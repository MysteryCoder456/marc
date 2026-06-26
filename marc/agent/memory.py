from datetime import datetime
from typing import TYPE_CHECKING, ClassVar, final

from anyio import Path
from langchain.agents import create_agent
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AnyMessage
from mem0 import AsyncMemoryClient
from pydantic import BaseModel

from marc.dirs import DIRS

if TYPE_CHECKING:
    from marc.chat.storage import ChatSession


@final
class UserMemory:
    USER_MEMORY_PATH = Path(DIRS.user_data_path / "USER.md")

    @classmethod
    async def write(cls, memory: str):
        await cls.USER_MEMORY_PATH.write_text(memory)

    @classmethod
    async def read(cls) -> str:
        return await cls.USER_MEMORY_PATH.read_text()


@final
class ShortTermMemory:
    TODAY_PATH = Path(DIRS.user_data_path / "TODAY.md")

    class ChatSummaryOutput(BaseModel):
        summary: list[str]

    class ReconciliationOutput(BaseModel):
        new_summaries: str

    @classmethod
    async def _generate_chat_summary(
        cls, messages: list[AnyMessage]
    ) -> list[str]:
        model = ChatAnthropic(
            model="claude-sonnet-4-6",  # pyright: ignore[reportCallIssue]
            effort="medium",
            temperature=0.4,
        )
        system_prompt = (
            "Summarize this conversation as bullet points for a daily activity log.\n\n"
            "Include:\n"
            "- Decisions made\n"
            "- Tasks completed\n"
            "- Specific outputs (text, code, commands, file paths)\n"
            "- Unresolved issues\n\n"
            "Omit:\n"
            "- Small talk\n"
            "- Failed attempts that were corrected\n"
            "- Specific details about the user (those are stored elsewhere)\n"
            "- Anything not worth recalling tomorrow"
        )
        agent = create_agent(
            model=model,
            system_prompt=system_prompt,
            response_format=cls.ChatSummaryOutput,
        )

        response = await agent.ainvoke(
            {
                "messages": messages
                + [{"role": "user", "content": "Summarize this conversation."}]
            }
        )
        return response["structured_response"].summary

    @classmethod
    async def _reconcile_summaries(
        cls, session: "ChatSession", summary: list[str]
    ) -> str:
        current_summaries = await cls.read()

        model = ChatAnthropic(
            model="claude-sonnet-4-6",  # pyright: ignore[reportCallIssue]
            effort="medium",
            temperature=0.4,
        )
        system_prompt = (
            "You maintain a daily activity log. Each session entry uses this format:\n\n"
            "**<session name>** (ID: <id>)\n"
            "- bullet\n\n"
            "Given the current log and a new session summary below, return the complete updated log.\n\n"
            "Rules:\n"
            "- Find the session by ID; replace its entry with the new summary.\n"
            "- If no matching ID exists, append the new entry.\n"
            "- When old and new facts conflict, the new summary wins.\n"
            "- Drop irrelevant or fully superseded facts.\n"
            "- Leave all other sessions exactly as they are.\n"
            "- If the current log is empty, return only the new entry.\n"
            "- Return the updated log only — no commentary.\n\n"
            f"Current log:\n\n{current_summaries}"
        )
        agent = create_agent(
            model=model,
            system_prompt=system_prompt,
            response_format=cls.ReconciliationOutput,
        )

        formatted_summary = "\n".join(f"- {line}" for line in summary)
        response = await agent.ainvoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": f"**{session.name}** (ID: {session.id})\n{formatted_summary}",
                    }
                ]
            }
        )
        return response["structured_response"].new_summaries

    @classmethod
    async def save(cls, session: "ChatSession"):
        summary = await cls._generate_chat_summary(session.messages)
        new_summaries = await cls._reconcile_summaries(session, summary)
        await cls.TODAY_PATH.write_text(new_summaries)

    @classmethod
    async def read(cls) -> str:
        if await cls.TODAY_PATH.exists():
            return await cls.TODAY_PATH.read_text()
        return "*(empty — nothing saved yet)*"

    @classmethod
    async def delete(cls):
        await cls.TODAY_PATH.unlink(missing_ok=True)


@final
class LongTermMemory:
    client: ClassVar[AsyncMemoryClient]

    @classmethod
    def init(cls):
        """
        Initialize Mem0 client. Meant to be called after environment
        variables have been loaded.
        """

        cls.client = AsyncMemoryClient()

    @classmethod
    async def should_dream(cls) -> bool:
        stm_path = ShortTermMemory.TODAY_PATH

        if not await stm_path.exists():
            # Nothing to consolidate
            return False

        stm_mtime = (await stm_path.stat()).st_mtime
        stm_changed = datetime.fromtimestamp(stm_mtime)

        today = datetime.now().date()
        today_midnight = datetime.combine(today, datetime.min.time())

        # Whether STM was changed yesterday
        return stm_changed < today_midnight

    @classmethod
    async def dream(cls):
        stm = await ShortTermMemory.read()
        await cls.client.add(
            f"Here's a summary of our conversations from yesterday. Disregard information specifically about the user; only remember conversation/task details.\n\n{stm}",
            app_id="cv.rehatsingh.marc",
        )
        await ShortTermMemory.delete()
