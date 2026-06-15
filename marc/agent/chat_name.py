from langchain.agents import create_agent
from langchain_core.messages import AnyMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel


class ChatNameOutput(BaseModel):
    name: str


async def generate_chat_name(messages: list[AnyMessage]) -> str:
    model = ChatOpenAI(model="gpt-5.4-nano", reasoning={"effort": "none"})
    agent = create_agent(
        model=model,
        system_prompt="Name the following chat conversation. Reply with only the name: at most 5 words, capturing the conversation's main topic.",
        response_format=ChatNameOutput,
    )

    response = await agent.ainvoke({"messages": messages})  # pyright: ignore[reportArgumentType]
    return response["structured_response"].name
