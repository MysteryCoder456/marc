from typing import override
from textual.app import App, ComposeResult
from textual.widgets import Digits, Footer, Header


class Smiley(Digits):
    def __init__(self) -> None:
        super().__init__("8)")


class MarcApp(App):
    @override
    def compose(self) -> ComposeResult:
        yield Header()
        yield Smiley()
        yield Footer()


if __name__ == "__main__":
    app = MarcApp()
    app.run()
