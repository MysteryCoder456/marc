import string
from typing import Any, final
from uuid import UUID, uuid4

from anyio import Path
from langchain_core.messages import (
    AnyMessage,
    messages_from_dict,
    messages_to_dict,
)
from pydantic import (
    BaseModel,
    Field,
    field_serializer,
    field_validator,
)

from marc.dirs import DIRS


class ChatSession(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    name: str | None = Field(default=None)
    messages: list[AnyMessage] = Field(default=[])

    @field_serializer("messages", mode="plain")
    def serialize_messages(self, value: list[AnyMessage]) -> list[dict]:
        return messages_to_dict(value)

    @field_validator("messages", mode="before", json_schema_input_type=dict)
    @classmethod
    def validate_messages(cls, value: Any) -> list[AnyMessage]:  # pyright: ignore[reportExplicitAny]
        if not isinstance(value, list):
            raise ValueError(
                f"`messages` must be a list of objects, not {type(value)}"
            )

        return messages_from_dict(value)  # pyright: ignore[reportReturnType]


@final
class ChatStorage:
    CHATS_PATH = Path(DIRS.user_data_path / "chats")

    @classmethod
    async def _ensure_paths(cls):
        await cls.CHATS_PATH.mkdir(exist_ok=True)

    @classmethod
    async def list_chats(cls) -> list[ChatSession]:
        """
        List all the chats stored on disk.

        Returns:
            A list of `ChatSession`s without their `messages` field populated.
        """

        await cls._ensure_paths()

        paths = [
            (path, (await path.stat()).st_ctime)
            async for path in cls.CHATS_PATH.glob("*.json")
        ]
        paths.sort(key=lambda pair: pair[1], reverse=True)
        return [
            ChatSession(
                id=UUID(path.stem.split("+")[0]),
                name=path.stem.split("+")[1].replace("_", " "),
            )
            for path, _ in paths
        ]

    @classmethod
    async def load_chat(cls, chat_id: UUID) -> ChatSession | None:
        """
        Loads a chat session from disk.

        Args:
            chat_id: Unique ID of the chat to load.

        Returns:
            The chat session with the specified ID, or `None` if no such chat
            exists.
        """

        await cls._ensure_paths()
        paths = cls.CHATS_PATH.glob(f"{chat_id}+*.json")

        if chat_path := await anext(paths):
            chat_json = await chat_path.read_text()
            return ChatSession.model_validate_json(chat_json)

        return None

    @classmethod
    async def save_chat(cls, chat: ChatSession):
        """
        Saves a chat session to disk.

        Args:
            chat: The chat session to save.
        """

        await cls._ensure_paths()

        # Sanitize chat name
        if chat.name:
            safe_name = "".join(
                ch
                for ch in chat.name
                if ch in string.printable and ch not in string.punctuation
            )
            for ch in string.whitespace:
                safe_name = safe_name.replace(ch, "_")

        else:
            safe_name = "Untitled"

        # Save to disk
        chat_path = cls.CHATS_PATH / f"{chat.id}+{safe_name}.json"
        chat_json = chat.model_dump_json()
        await chat_path.write_text(chat_json)
