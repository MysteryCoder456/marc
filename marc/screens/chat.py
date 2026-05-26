from pathlib import Path
from typing import final, override

from langchain_core.messages import AnyMessage
from textual import on, work
from textual.app import ComposeResult, RenderResult
from textual.containers import VerticalGroup, VerticalScroll
from textual.reactive import reactive
from textual.screen import Screen
from textual.widgets import Footer, Header, Input, Static
from textual.worker import Worker

from marc.agent import RuntimeContext, create_new_agent


@final
class ChatMessage(Static):
    def __init__(self, msg: AnyMessage) -> None:
        super().__init__(id=f"msg-{msg.id}", classes="message", markup=False)

        self.msg = msg
        self.border_title = msg.type.capitalize()

    @override
    def render(self) -> RenderResult:
        return str(self.msg.content)


@final
class ChatScreen(Screen):
    CSS_PATH = "chat.tcss"

    messages: reactive[list[AnyMessage]] = reactive([])
    added_messages: reactive[set[str]] = reactive(set())

    def __init__(self) -> None:
        super().__init__()

        self.agent = create_new_agent()

    def on_mount(self):
        # Focus input
        self.query_one("#chat-input").focus()

    def watch_messages(self, msgs: list[AnyMessage]):
        new_msgs = [msg for msg in msgs if msg.id not in self.added_messages]
        if not new_msgs:
            return

        self.added_messages.update([msg.id for msg in new_msgs])  # pyright: ignore[reportArgumentType]
        self.mutate_reactive(ChatScreen.added_messages)

        msg_widgets = [ChatMessage(msg) for msg in new_msgs]
        msg_container = self.query_one("#messages")
        msg_container.mount_all(msg_widgets)

    @on(Input.Submitted, "#chat-input")
    def send_message(self, event: Input.Submitted):
        msg = event.value
        if not msg:
            return

        # Reset input
        event.control.value = ""
        event.control.focus()

        self.send_message_to_agent(msg)

    @work
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

        yield VerticalScroll(id="messages")

        with VerticalGroup(id="bottom-dock"):
            yield Input(placeholder="Chat", id="chat-input")
            yield Footer()
