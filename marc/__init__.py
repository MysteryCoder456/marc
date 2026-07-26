from collections.abc import Iterable
from typing import final, override
from uuid import UUID

from dotenv import load_dotenv
from textual import on
from textual.app import App, SystemCommand
from textual.command import CommandPalette
from textual.screen import Screen
from textual.widgets import Static

from marc.agent.memory import LongTermMemory
from marc.chat.providers import ChatListProvider
from marc.chat.screen import ChatScreen
from marc.dream.screen import DreamModeScreen
from marc.work.screen import WorkModeScreen


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
    SCREENS = {
        "work_mode": WorkModeScreen,
    }

    current_chat: UUID | None = None

    def open_chat(self, chat_id: UUID | None = None):
        screen = ChatScreen(chat_id)

        if len(self.screen_stack) <= 1:
            self.push_screen(screen)
        else:
            self.switch_screen(screen)

    async def on_mount(self):
        LongTermMemory.init()

        if await LongTermMemory.should_dream():
            self.push_screen(DreamModeScreen())
        else:
            self.open_chat()

    def action_new_chat(self):
        self.open_chat()

    def action_switch_chat(self):
        self.push_screen(
            CommandPalette([ChatListProvider], placeholder="Search for chats…")
        )

    @on(DreamModeScreen.FinishedDreaming)
    def on_finished_dreaming(self, _event: DreamModeScreen.FinishedDreaming):
        self.open_chat()

    @on(ChatScreen.Loaded)
    def on_chat_loaded(self, event: ChatScreen.Loaded):
        self.current_chat = event.chat_id

    @override
    def get_system_commands(self, screen: Screen) -> Iterable[SystemCommand]:
        if len(self.screen_stack) > 1 and isinstance(
            self.screen_stack[-2], WorkModeScreen
        ):
            return

        yield from super().get_system_commands(screen)
        yield SystemCommand(
            "New Chat",
            "Start a fresh conversation with Marc.",
            self.action_new_chat,
        )
        yield SystemCommand(
            "Chats",
            "Go to another conversation you've had with Marc",
            self.action_switch_chat,
        )


load_dotenv()
app = MarcApp()
