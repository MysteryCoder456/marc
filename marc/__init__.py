from typing import final

from dotenv import load_dotenv
from textual.app import App
from textual.widgets import Static

from .screens import ChatScreen


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
class MarcApp(App):
    TITLE = "Marc"
    SCREENS = {"chat": ChatScreen}

    def on_mount(self):
        self.push_screen("chat")


load_dotenv()
app = MarcApp()

