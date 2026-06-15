from typing import final

from anyio import Path
from langchain.agents import create_agent
from langchain_core.messages import AnyMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from marc.chat.storage import ChatSession
from marc.dirs import DIRS


class UserMemory:
    # TODO: Migrate from agent.py
    ...


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
        model = ChatOpenAI(model="gpt-5.4-nano", reasoning={"effort": "none"})
        agent = create_agent(
            model=model,
            system_prompt="Summarize this conversation as bullet points for a daily activity log. Include: decisions made, tasks completed, specific outputs (text, code, commands, file paths), unresolved issues. Omit small talk, failed attempts that were corrected, and anything not worth recalling tomorrow.",
            response_format=cls.ChatSummaryOutput,
        )

        response = await agent.ainvoke({"messages": messages})  # pyright: ignore[reportArgumentType]
        return response["structured_response"].summary

    @classmethod
    async def _reconcile_summaries(
        cls, session: ChatSession, summary: list[str]
    ) -> str:
        current_summaries = await cls.read()

        model = ChatOpenAI(model="gpt-5.4-nano", reasoning={"effort": "none"})
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
                        "role": "human",
                        "content": f"**{session.name}** (ID: {session.id})\n{formatted_summary}",
                    }
                ]
            }
        )
        return response["structured_response"].new_summaries

    @classmethod
    async def save(cls, session: ChatSession):
        # Generate a summary of the session
        summary = await cls._generate_chat_summary(session.messages)

        # Reconcile with existing session facts
        new_summaries = await cls._reconcile_summaries(session, summary)

        # Write to disk
        await cls.TODAY_PATH.write_text(new_summaries)

    @classmethod
    async def read(cls) -> str:
        if await cls.TODAY_PATH.exists():
            return await cls.TODAY_PATH.read_text()
        return "*(empty — nothing saved yet)*"
