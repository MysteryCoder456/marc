import datetime
import io
from base64 import b64encode
from typing import Annotated

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import InjectedToolCallId, tool
from langchain_core.messages import (
    ImageContentBlock,
    ToolMessage,
)
from langchain_core.prompts.chat import MessageLike
from langgraph.checkpoint.memory import InMemorySaver
from mss import MSS
from PIL import Image


@tool
def get_time() -> str:
    """
    Get the current time.
    """

    now = datetime.datetime.now()
    return str(now)


@tool
def take_screenshot(
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> ToolMessage:
    """
    Takes a screenshot of the user's desktop and returns it as a base64
    encoded image.
    """

    with MSS() as sct:
        # Grab screenshot
        monitor = sct.monitors[1]
        sct_img = sct.grab(monitor)

    # Save image data into a buffer
    img = Image.frombytes("RGB", sct_img.size, sct_img.rgb)  # pyright: ignore[reportUnknownArgumentType]
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    img_bytes = buf.getvalue()

    return ToolMessage(
        content_blocks=[
            ImageContentBlock(
                type="image",
                base64=b64encode(img_bytes).decode("utf-8"),
                mime_type="image/jpeg",
            )
        ],
        tool_call_id=tool_call_id,
    )


def main():
    if not load_dotenv():
        print("Oops... Couldn't load .env file.")

    memory = InMemorySaver()
    agent = create_agent(
        model="anthropic:claude-haiku-4-5",
        system_prompt="You are a helpful computer-use assistant. You must complete tasks that the user gives you by using the tools at your disposal.",
        tools=[get_time, take_screenshot],
        checkpointer=memory,
    )

    while True:
        try:
            user_input = input("\n> ")

            response = agent.stream(
                {"messages": [{"role": "user", "content": user_input}]},
                {"configurable": {"thread_id": "thread-1"}},
                stream_mode="values",
            )
            for chunk in response:
                chunk_msg: MessageLike = chunk["messages"][-1]
                print()
                chunk_msg.pretty_print()

        except KeyboardInterrupt:
            break


if __name__ == "__main__":
    main()
