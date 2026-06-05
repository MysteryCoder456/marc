from collections.abc import Iterable
from functools import partial
from pathlib import Path
from typing import final, override
from uuid import UUID

from dotenv import load_dotenv
from textual import on
from textual.app import App, SystemCommand
from textual.command import CommandPalette, DiscoveryHit, Hit, Hits, Provider
from textual.screen import Screen
from textual.widgets import Static

from .dirs import CHATS_PATH, ensure_paths
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
class ChatListProvider(Provider):
    # TODO: show chat names and last modified date

    chat_ids: list[UUID] = []

    def list_chat_paths(self) -> list[Path]:
        paths = list(CHATS_PATH.glob("*.json"))
        paths.sort(key=lambda p: p.stat().st_ctime, reverse=True)
        return paths

    @override
    async def startup(self) -> None:
        worker = self.app.run_worker(self.list_chat_paths, thread=True)
        chat_paths = await worker.wait()
        self.chat_ids = [UUID(path.stem) for path in chat_paths]

    @override
    async def search(self, query: str) -> Hits:
        assert isinstance(self.app, MarcApp)
        matcher = self.matcher(query)

        for chat_id in self.chat_ids:
            if chat_id == self.app.current_chat:
                continue

            score = matcher.match(chat_id.hex)
            if score > 0:
                yield Hit(
                    score,
                    matcher.highlight(chat_id.hex),
                    partial(self.app.open_chat, chat_id),
                )

    @override
    async def discover(self) -> Hits:
        assert isinstance(self.app, MarcApp)

        for chat_id in self.chat_ids:
            if chat_id == self.app.current_chat:
                continue

            yield DiscoveryHit(
                chat_id.hex,
                partial(self.app.open_chat, chat_id),
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

    def on_mount(self):
        ensure_paths()
        self.open_chat()

    def action_new_chat(self):
        self.open_chat()

    def action_switch_chat(self):
        self.push_screen(
            CommandPalette([ChatListProvider], placeholder="Search for chats…")
        )

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
