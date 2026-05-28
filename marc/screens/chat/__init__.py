from pathlib import Path
from typing import final, override

from langchain_core.messages import AnyMessage
from textual import on, work
from textual.app import ComposeResult
from textual.containers import VerticalGroup, VerticalScroll
from textual.reactive import reactive
from textual.screen import Screen
from textual.widgets import Footer, Header, Input
from textual.worker import Worker, WorkerState

from marc.agent import RuntimeContext, create_new_agent

from .indicator import RunningIndicator
from .message import ChatMessage


@final
class ChatScreen(Screen):
    CSS_PATH = "styles.tcss"

    messages: reactive[list[AnyMessage]] = reactive([])
    added_messages: reactive[set[str]] = reactive(set())
    is_agent_running = reactive(False)

    def __init__(self) -> None:
        super().__init__()

        self.agent = create_new_agent()

    def on_mount(self):
        # Focus input
        self.query_one("#chat-input").focus()

    def scroll_to_end(self):
        scroller = self.query_one("#chat-scroll-area")
        scroller.scroll_end(animate=False)

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
        response = self.agent.astream(
            {"messages": [{"role": "user", "content": msg}]},
            {"configurable": {"thread_id": "thread-1"}},
            stream_mode="values",
            context=RuntimeContext(cwd=Path.cwd()),
        )
        latest_seen_idx = 0

        async for chunk in response:
            chunk_msgs = chunk["messages"]
            new_msgs = chunk_msgs[latest_seen_idx:]

            self.messages.extend(new_msgs)
            self.mutate_reactive(ChatScreen.messages)

            latest_seen_idx = len(chunk_msgs)

    @override
    def compose(self) -> ComposeResult:
        yield Header()

        with VerticalScroll(id="chat-scroll-area"):
            yield VerticalGroup(id="messages")
            yield RunningIndicator()

        with VerticalGroup(id="bottom-dock"):
            yield Input(placeholder="Chat", id="chat-input")
            yield Footer()
