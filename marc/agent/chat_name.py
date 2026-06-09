from langchain.agents import create_agent
from langchain_core.messages import AnyMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

SYSTEM_PROMPT = """
You are a helpful assistant that generates concise and descriptive names for chat conversations based on their content. Given a list of messages from a chat, your task is to analyze the conversation and come up with an appropriate name that captures the main topic or theme of the discussion. The name should be brief, ideally no more than 5 words, and should accurately reflect the essence of the conversation.
"""


class ChatNameOutput(BaseModel):
    name: str


async def generate_chat_name(messages: list[AnyMessage]) -> str:
    model = ChatOpenAI(model="gpt-5.4-nano", reasoning={"effort": "none"})
    agent = create_agent(
        model=model,
        system_prompt=SYSTEM_PROMPT,
        response_format=ChatNameOutput,
    )

    response = await agent.ainvoke({"messages": messages})  # pyright: ignore[reportArgumentType]
    return response["structured_response"].name
