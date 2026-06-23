from collections.abc import Iterable
from functools import partial
from typing import final, override
from uuid import UUID

from dotenv import load_dotenv
from textual import on
from textual.app import App, SystemCommand
from textual.command import CommandPalette, DiscoveryHit, Hit, Hits, Provider
from textual.screen import Screen
from textual.widgets import Static

from marc.agent.memory import LongTermMemory
from marc.chat.screen import ChatScreen
from marc.chat.storage import ChatSession, ChatStorage
from marc.dream.screen import DreamModeScreen


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
class ChatListProvider(Provider):
    chats: list[ChatSession] = []

    @override
    async def startup(self) -> None:
        self.chats = await ChatStorage.list_chats()

    @override
    async def search(self, query: str) -> Hits:
        assert isinstance(self.app, MarcApp)
        matcher = self.matcher(query)

        for chat in self.chats:
            if chat.id == self.app.current_chat:
                continue

            score = matcher.match(chat.name or "Untitled")
            if score > 0:
                yield Hit(
                    score,
                    matcher.highlight(chat.name or "Untitled"),
                    partial(self.app.open_chat, chat.id),
                )

    @override
    async def discover(self) -> Hits:
        assert isinstance(self.app, MarcApp)

        for chat in self.chats:
            if chat.id == self.app.current_chat:
                continue

            yield DiscoveryHit(
                chat.name or "Untitled",
                partial(self.app.open_chat, chat.id),
            )


@final
class MarcApp(App):
    TITLE = "Marc"

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
