from typing import final, override

from textual import on
from textual.app import ComposeResult
from textual.containers import VerticalGroup, VerticalScroll
from textual.reactive import reactive
from textual.screen import Screen
from textual.widgets import Footer, Header, Input, Label

from marc.agent import create_new_agent


@final
class ChatScreen(Screen):
    CSS_PATH = "chat.tcss"

    messages: reactive[list[str]] = reactive([])

    def __init__(self) -> None:
        super().__init__()

        self.agent = create_new_agent()

    def on_mount(self):
        # Focus input
        self.query_one("#chat-input").focus()

    def watch_messages(self, msgs: list[str]):
        # HACK: remove and readd all messages because no way to identify new ones

        labels = [Label(msg) for msg in msgs]
        if not labels:
            return

        msg_container = self.query_one("#messages")
        msg_container.remove_children()
        msg_container.mount_all(labels)

    @on(Input.Submitted, "#chat-input")
    def send_message(self, event: Input.Submitted):
        msg = event.value
        if not msg:
            return

        # Reset input
        event.control.value = ""
        event.control.focus()

        # TODO: send message to agent
        self.messages.append(msg)
        self.mutate_reactive(ChatScreen.messages)

    @override
    def compose(self) -> ComposeResult:
        yield Header()

        with VerticalScroll(id="messages"):
            for msg in self.messages:
                yield Label(msg)

        with VerticalGroup(id="bottom-dock"):
            yield Input(placeholder="Chat", id="chat-input")
            yield Footer()
