import asyncio
import json
from datetime import datetime
from typing import TYPE_CHECKING, ClassVar, final

import aiofiles
from anyio import Path
from langchain.agents import create_agent
from langchain_core.messages import AnyMessage
from langchain_openai import ChatOpenAI
from mem0 import AsyncMemoryClient
from pydantic import BaseModel

from marc.dirs import DIRS

if TYPE_CHECKING:
    from marc.chat.storage import ChatSession


def get_model() -> ChatOpenAI:
    return ChatOpenAI(
        model="gpt-5.6-luna",
        reasoning={"effort": "none"},
    )


@final
class UserMemory:
    USER_MEMORY_PATH = Path(DIRS.user_data_path / "USER.md")

    @classmethod
    async def write(cls, memory: str):
        await cls.USER_MEMORY_PATH.write_text(memory)

    @classmethod
    async def read(cls) -> str:
        if await cls.USER_MEMORY_PATH.exists():
            return await cls.USER_MEMORY_PATH.read_text()
        return "*(empty — nothing saved yet)*"


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
            "- Anything not worth recalling tomorrow\n\n"
            "Be consistent and matter-of-fact — stick to what happened, without "
            "embellishment or creative phrasing."
        )
        agent = create_agent(
            model=get_model(),
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
            "Apply these rules consistently and deterministically — don't vary "
            "formatting or phrasing between entries.\n\n"
            f"Current log:\n\n{current_summaries}"
        )
        agent = create_agent(
            model=get_model(),
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
    INDEX_PATH = Path(DIRS.user_data_path / "LTM_INDEX.md")

    client: ClassVar[AsyncMemoryClient]

    class IndexOutput(BaseModel):
        updated_index: dict[str, list[str]]

    @classmethod
    def init(cls):
        """
        Initialize Mem0 client. Meant to be called after environment
        variables have been loaded.
        """

        cls.client = AsyncMemoryClient()

    @classmethod
    async def read_index(cls) -> str:
        if await cls.INDEX_PATH.exists():
            return await cls.INDEX_PATH.read_text()
        return "*(empty — nothing saved yet)*"

    @classmethod
    async def write_index(cls, index: dict[str, list[str]]):
        async with aiofiles.open(cls.INDEX_PATH, "w") as f:
            for category, items in index.items():
                await f.write(f"# {category}\n\n")
                bulleted_items = [f"- {item}" for item in items]
                await f.write("\n".join(bulleted_items))
                await f.write("\n\n")

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
        app_id = "com.rehatsingh.marc"
        run_tag = f"dream-{datetime.now().isoformat()}"

        # Save to store
        await cls.client.add(
            f"Here's a summary of our conversations from yesterday. Disregard information specifically about the user; only remember conversation/task details.\n\n{stm}",
            app_id=app_id,
            run_id=run_tag,
        )

        # HACK: use the events API to know when memory inference is complete
        wait = 5
        while True:
            await asyncio.sleep(wait)
            memories_response = await cls.client.get_all(
                filters={"app_id": app_id, "run_id": run_tag}
            )
            if memories_response.get("count", 0) > 0:
                break
            wait *= 1.25
        delta_memories = memories_response["results"]

        # Update index
        index = await cls.read_index()
        index_prompt = (
            "You maintain an index of what is stored in Long Term Memory (LTM). "
            "The index is a table of contents, not the memories themselves: each "
            "entry names a topic so a reader knows a memory about it exists, "
            "without restating what the memory says.\n\n"
            "Given the current index and the memories that were added or changed "
            "below, return the complete updated index as categories mapped to "
            "their topics. Each memory arrives as its text plus `categories`, "
            "which the LTM storage system may have assigned.\n\n"
            "Rules:\n"
            "- Turn each memory into a topic entry, and file it under the "
            "categories it came with. A memory carrying several categories is "
            "filed under each of them.\n"
            "- Infer a category only when `categories` is empty: reuse an "
            "existing category from the index where one fits, and create a new "
            "one only when none does.\n"
            "- Categories are broad life domains — technology, finance, "
            "business, career, health, travel and the like — not narrow labels "
            "for one memory. An inferred category must match that breadth.\n"
            "- Fold a new topic into an existing entry when they cover the same "
            "thing; add a separate entry only when it is genuinely distinct.\n"
            "- Keep entries short — a noun phrase naming the subject, not a "
            "sentence and not the memory's content.\n"
            "- Leave categories and entries untouched by the new topics exactly "
            "as they are; never drop an existing entry.\n"
            "- Return the updated index only — no commentary.\n\n"
            "Apply these rules consistently and deterministically — don't vary "
            "category names or phrasing between runs.\n\n"
            f"Current index:\n\n{index}"
        )
        index_agent = create_agent(
            model=get_model(),
            system_prompt=index_prompt,
            response_format=cls.IndexOutput,
        )

        sanitized_memories = [
            {
                "memory": mem.get("memory"),
                "categories": mem.get("categories", []),
            }
            for mem in delta_memories
        ]
        index_response = await index_agent.ainvoke(
            {
                "messages": [
                    {"role": "user", "content": json.dumps(sanitized_memories)}
                ]
            }
        )
        updated_index: dict[str, list[str]] = index_response[
            "structured_response"
        ].updated_index
        await cls.write_index(updated_index)

        await ShortTermMemory.delete()
