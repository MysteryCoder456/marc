from typing import final

from dotenv import load_dotenv
from textual.app import App
from textual.widgets import Static

from .dirs import ensure_paths
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

    def on_mount(self):
        ensure_paths()
        self.push_screen(ChatScreen())


load_dotenv()
app = MarcApp()
