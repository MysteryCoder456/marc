from typing import override

from textual.app import App, ComposeResult
from textual.widgets import Footer, Header, Static


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


class MarcApp(App):
    @override
    def compose(self) -> ComposeResult:
        yield Header()
        yield Smiley()
        yield Footer()


if __name__ == "__main__":
    app = MarcApp()
    app.run()
