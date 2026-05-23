from typing import final, override

from textual.app import App, ComposeResult
from textual.containers import Container, VerticalGroup
from textual.reactive import reactive
from textual.widget import Widget
from textual.widgets import Footer, Header, Input, Label, Static


class Smiley(Static):
    def __init__(self) -> None:
        super().__init__(
            """        ─╮
╶──╴     │
     ═   │
╶──╴     │
        ─╯
"""
        )


@final
class Name(Widget):
    who = reactive("Hello", recompose=True)

    @override
    def compose(self) -> ComposeResult:
        with Container():
            yield Label(self.who)


@final
class MarcApp(App):
    CSS_PATH = "main.tcss"

    def on_input_changed(self, event: Input.Changed):
        self.query_one(Name).who = event.value

    @override
    def compose(self) -> ComposeResult:
        yield Header()

        yield Name()

        with VerticalGroup(id="bottom-dock"):
            yield Input(placeholder="Chat", id="chat-input")
            yield Footer()


app = MarcApp()

