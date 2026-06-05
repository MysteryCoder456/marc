import json
from pathlib import Path
from typing import final, override
from uuid import UUID, uuid4

import aiofiles
from langchain_core.messages import (
    AnyMessage,
    messages_from_dict,
    messages_to_dict,
)
from textual import on, work
from textual.app import ComposeResult
from textual.containers import VerticalGroup, VerticalScroll
from textual.reactive import reactive
from textual.screen import Screen
from textual.widgets import Footer, Input
from textual.worker import Worker, WorkerState

from marc.agent import RuntimeContext, create_new_agent
from marc.dirs import CHATS_PATH

from .indicator import RunningIndicator
from .message import ChatMessage


@final
class ChatScreen(Screen):
    CSS_PATH = "styles.tcss"

    messages: reactive[list[AnyMessage]] = reactive([])
    added_messages: reactive[set[str]] = reactive(set())
    is_agent_running = reactive(False)

    def __init__(self, chat_id: UUID | None = None) -> None:
        super().__init__()

        self.chat_id = chat_id
        self.agent = create_new_agent()
        self.is_context_loaded = True

    async def load_chat(self, chat_id: UUID):
        # Load previous messages
        chat_path = CHATS_PATH / f"{chat_id}.json"
        async with aiofiles.open(chat_path, "r") as f:
            messages_json = await f.read()
            messages_dict = json.loads(messages_json)
            self.messages = messages_from_dict(messages_dict)  # pyright: ignore[reportAttributeAccessIssue]

        self.log("Loaded chat", chat_id, "from disk.")
        self.is_context_loaded = False

    async def save_chat(self, chat_id: UUID):
        if not (self.messages and self.is_context_loaded):
            return

        # Save chat to disk
        chat_path = CHATS_PATH / f"{chat_id}.json"
        async with aiofiles.open(chat_path, "w") as f:
            messages_dict = messages_to_dict(self.messages)
            messages_json = json.dumps(messages_dict)
            await f.write(messages_json)

        self.log("Saved chat", chat_id, "to disk.")

    def scroll_to_end(self):
        scroller = self.query_one("#chat-scroll-area")
        scroller.scroll_end(animate=False)

    # ================ ↓ TEXTUAL FUNCTIONS ↓ ================

    async def on_mount(self):
        if self.chat_id:
            # Opening existing chat
            await self.load_chat(self.chat_id)
        else:
            # Opening new chat
            self.chat_id = uuid4()

        # Focus input
        self.query_one("#chat-input").focus()

    async def on_unmount(self):
        if self.chat_id:
            await self.save_chat(self.chat_id)

    async def watch_messages(self, msgs: list[AnyMessage]):
        # Find newly added messages
        new_msgs = [msg for msg in msgs if msg.id not in self.added_messages]
        if not new_msgs:
            return

        # Update state
        self.added_messages.update([msg.id for msg in new_msgs])  # pyright: ignore[reportArgumentType]
        self.mutate_reactive(ChatScreen.added_messages)

        # Mount new message widgets
        msg_widgets = [ChatMessage(msg) for msg in new_msgs]
        msg_container = self.query_one("#messages")
        await msg_container.mount_all(msg_widgets)

        self.scroll_to_end()

    def watch_is_agent_running(self, running: bool):
        indicator = self.query_one(RunningIndicator)

        if running:
            indicator.show()
        else:
            indicator.hide()

    @on(Input.Submitted, "#chat-input")
    def on_chat_input_submitted(self, event: Input.Submitted):
        msg = event.value
        if not msg:
            return

        # Reset input
        event.control.value = ""
        event.control.focus()

        self.send_message_to_agent(msg)

    @on(Worker.StateChanged)
    def on_agent_worker_state_changed(self, event: Worker.StateChanged):
        if event.worker.name != "agent_message":
            return

        match event.state:
            case WorkerState.PENDING:
                pass

            case WorkerState.RUNNING:
                self.is_agent_running = True

            case _:
                self.is_agent_running = False
                self.scroll_to_end()

    @work(name="agent_message", exclusive=True)
    async def send_message_to_agent(self, msg: str):
        query_messages: list[AnyMessage | dict[str, str]] = [
            {"role": "user", "content": msg}
        ]

        if not self.is_context_loaded:
            # Inject session's previous messages into context
            query_messages = self.messages + query_messages
            self.is_context_loaded = True

        response = self.agent.astream(
            {"messages": query_messages},
            {"configurable": {"thread_id": self.chat_id}},
            stream_mode="values",
            context=RuntimeContext(cwd=Path.cwd()),
        )
        next_msg_idx = len(self.messages)

        async for chunk in response:
            chunk_msgs = chunk["messages"]
            new_msgs = chunk_msgs[next_msg_idx:]

            self.messages.extend(new_msgs)
            self.mutate_reactive(ChatScreen.messages)

            next_msg_idx = len(chunk_msgs)

    @override
    def compose(self) -> ComposeResult:
        with VerticalScroll(id="chat-scroll-area"):
            yield VerticalGroup(id="messages")
            yield RunningIndicator()

        with VerticalGroup(id="bottom-dock"):
            yield Input(placeholder="Chat", id="chat-input")
            yield Footer()
