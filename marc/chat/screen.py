import asyncio
from typing import final, override
from uuid import UUID

from langchain_core.messages import (
    AnyMessage,
)
from langchain_core.runnables import Runnable
from textual import on, work
from textual.app import ComposeResult
from textual.containers import VerticalGroup, VerticalScroll
from textual.message import Message
from textual.reactive import reactive
from textual.screen import Screen
from textual.widgets import Footer, Input
from textual.worker import Worker, WorkerState

from marc.agent import RuntimeContext, create_runtime_context, create_new_agent
from marc.agent.chat_name import generate_chat_name
from marc.agent.memory import ShortTermMemory

from .indicator import RunningIndicator
from .message import ChatMessage
from .storage import ChatSession, ChatStorage


@final
class ChatScreen(Screen):
    @final
    class Loaded(Message):
        def __init__(self, chat_id: UUID) -> None:
            super().__init__()
            self.chat_id = chat_id

    CSS_PATH = "styles.tcss"

    session = reactive(ChatSession, init=False)
    is_agent_running = reactive(False)

    def __init__(self, chat_id: UUID | None = None) -> None:
        super().__init__()

        self.chat_id = chat_id
        self.is_context_loaded = True
        self.added_messages: set[str] = set()
        self.agent_runtime: RuntimeContext

        self.agent: Runnable

    def scroll_to_end(self):
        scroller = self.query_one("#chat-scroll-area")
        scroller.scroll_end(animate=False)

    # ================ ↓ TEXTUAL FUNCTIONS ↓ ================

    async def on_mount(self):
        # Initialize agent
        self.agent = await create_new_agent()
        self.agent_runtime = await create_runtime_context()

        if self.chat_id and (ses := await ChatStorage.load_chat(self.chat_id)):
            # Open existing chat
            self.session = ses
            self.is_context_loaded = False

        else:
            # Open new chat
            self.chat_id = self.session.id

        self.app.post_message(ChatScreen.Loaded(self.chat_id))

        # Focus input
        self.query_one("#chat-input").focus()

    async def on_unmount(self):
        # Don't do anything if conversation state hasn't changed
        if not (
            self.chat_id and self.session.messages and self.is_context_loaded
        ):
            return

        # Save chat and STM
        save_coro = ChatStorage.save_chat(self.session)
        stm_coro = ShortTermMemory.save(self.session)
        await asyncio.gather(save_coro, stm_coro)

    async def watch_session(self, session: ChatSession):
        # Find newly added messages
        new_msgs = [
            msg
            for msg in session.messages
            if msg.id not in self.added_messages
        ]
        if not new_msgs:
            return
        self.added_messages.update([msg.id for msg in new_msgs])  # pyright: ignore[reportArgumentType]

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
        # Construct query
        query_messages: list[AnyMessage | dict[str, str]] = [
            {"role": "user", "content": msg}
        ]
        if not self.is_context_loaded:
            # Inject session's previous messages into context
            query_messages = self.session.messages + query_messages
            self.is_context_loaded = True

        # Stream agent responses
        next_msg_idx = len(self.session.messages)
        response = self.agent.astream(
            {"messages": query_messages},
            {"configurable": {"thread_id": self.chat_id}},
            stream_mode="values",
            context=self.agent_runtime,
        )
        async for chunk in response:
            chunk_msgs = chunk["messages"]
            new_msgs = chunk_msgs[next_msg_idx:]

            self.session.messages.extend(new_msgs)
            self.mutate_reactive(ChatScreen.session)  # pyright: ignore[reportArgumentType]

            next_msg_idx = len(chunk_msgs)

        # Generate a name for this session
        if not self.session.name:

            async def name_work():
                self.session.name = await generate_chat_name(
                    self.session.messages
                )

            self.run_worker(name_work())

    @override
    def compose(self) -> ComposeResult:
        with VerticalScroll(id="chat-scroll-area"):
            yield VerticalGroup(id="messages")
            yield RunningIndicator()

        with VerticalGroup(id="bottom-dock"):
            yield Input(placeholder="Chat", id="chat-input")
            yield Footer()
