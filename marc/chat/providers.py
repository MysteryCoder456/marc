from functools import partial
from typing import Protocol, final, override, runtime_checkable
from uuid import UUID

from textual.command import DiscoveryHit, Hit, Hits, Provider
from textual.message import Message

from marc.agent import get_available_skills
from marc.chat.storage import ChatSession, ChatStorage


@runtime_checkable
class MarcAppStub(Protocol):
    current_chat: UUID | None

    def open_chat(self, chat_id: UUID | None = None): ...


@final
class ChatListProvider(Provider):
    chats: list[ChatSession] = []

    @override
    async def startup(self) -> None:
        self.chats = await ChatStorage.list_chats()

    @override
    async def search(self, query: str) -> Hits:
        assert isinstance(self.app, MarcAppStub)
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
        assert isinstance(self.app, MarcAppStub)

        for chat in self.chats:
            if chat.id == self.app.current_chat:
                continue

            yield DiscoveryHit(
                chat.name or "Untitled",
                partial(self.app.open_chat, chat.id),
            )


@final
class SkillListProvider(Provider):
    @final
    class SkillSelected(Message):
        def __init__(self, skill_name: str) -> None:
            super().__init__()
            self.skill_name = skill_name

    def _select_skill(self, skill_name: str):
        self.screen.post_message(SkillListProvider.SkillSelected(skill_name))

    @override
    async def search(self, query: str) -> Hits:
        matcher = self.matcher(query)

        for skill_name in get_available_skills():
            score = matcher.match(skill_name)
            if score > 0:
                yield Hit(
                    score,
                    matcher.highlight(skill_name),
                    partial(self._select_skill, skill_name),
                )

    @override
    async def discover(self) -> Hits:
        for skill_name in get_available_skills():
            yield DiscoveryHit(
                skill_name,
                partial(self._select_skill, skill_name),
            )
