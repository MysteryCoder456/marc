import io
import platform
import time
from base64 import b64encode
from copy import copy
from enum import Enum
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
from pynput.keyboard import Controller as KeyboardController
from pynput.keyboard import Key
from pynput.mouse import Button
from pynput.mouse import Controller as MouseController


class ModifierKey(Enum):
    alt = "alt"
    backspace = "backspace"
    caps_lock = "caps_lock"
    cmd = "cmd"
    ctrl = "ctrl"
    delete = "delete"
    down = "down"
    end = "end"
    enter = "enter"
    esc = "esc"
    f1 = "f1"
    f2 = "f2"
    f3 = "f3"
    f4 = "f4"
    f5 = "f5"
    f6 = "f6"
    f7 = "f7"
    f8 = "f8"
    f9 = "f9"
    f10 = "f10"
    f11 = "f11"
    f12 = "f12"
    f13 = "f13"
    f14 = "f14"
    f15 = "f15"
    f16 = "f16"
    f17 = "f17"
    f18 = "f18"
    f19 = "f19"
    f20 = "f20"
    home = "home"
    left = "left"
    page_down = "page_down"
    page_up = "page_up"
    right = "right"
    shift = "shift"
    space = "space"
    tab = "tab"
    up = "up"
    media_play_pause = "media_play_pause"
    media_volume_mute = "media_volume_mute"
    media_volume_down = "media_volume_down"
    media_volume_up = "media_volume_up"
    media_previous = "media_previous"
    media_next = "media_next"


MODIFIER_KEY_MAP = {
    ModifierKey.alt: Key.alt,
    ModifierKey.backspace: Key.backspace,
    ModifierKey.caps_lock: Key.caps_lock,
    ModifierKey.cmd: Key.cmd,
    ModifierKey.ctrl: Key.ctrl,
    ModifierKey.delete: Key.delete,
    ModifierKey.down: Key.down,
    ModifierKey.end: Key.end,
    ModifierKey.enter: Key.enter,
    ModifierKey.esc: Key.esc,
    ModifierKey.f1: Key.f1,
    ModifierKey.f2: Key.f2,
    ModifierKey.f3: Key.f3,
    ModifierKey.f4: Key.f4,
    ModifierKey.f5: Key.f5,
    ModifierKey.f6: Key.f6,
    ModifierKey.f7: Key.f7,
    ModifierKey.f8: Key.f8,
    ModifierKey.f9: Key.f9,
    ModifierKey.f10: Key.f10,
    ModifierKey.f11: Key.f11,
    ModifierKey.f12: Key.f12,
    ModifierKey.f13: Key.f13,
    ModifierKey.f14: Key.f14,
    ModifierKey.f15: Key.f15,
    ModifierKey.f16: Key.f16,
    ModifierKey.f17: Key.f17,
    ModifierKey.f18: Key.f18,
    ModifierKey.f19: Key.f19,
    ModifierKey.f20: Key.f20,
    ModifierKey.home: Key.home,
    ModifierKey.left: Key.left,
    ModifierKey.page_down: Key.page_down,
    ModifierKey.page_up: Key.page_up,
    ModifierKey.right: Key.right,
    ModifierKey.shift: Key.shift,
    ModifierKey.space: Key.space,
    ModifierKey.tab: Key.tab,
    ModifierKey.up: Key.up,
    ModifierKey.media_play_pause: Key.media_play_pause,
    ModifierKey.media_volume_mute: Key.media_volume_mute,
    ModifierKey.media_volume_down: Key.media_volume_down,
    ModifierKey.media_volume_up: Key.media_volume_up,
    ModifierKey.media_previous: Key.media_previous,
}

SYSTEM_PROMPT = """# Computer Use Prompt

You are a computer-use assistant. You must help the user achieve various tasks on their computer using the tools at your disposal.

Respond to all queries in a concise manner.

## Operating System Specific Instructions

You should find out what OS the user is using before attempting to use keyboard shortcuts.

### Mac:

- Use the keyboard shortcut `ctrl + up` to use App Exposé, which shows all currently open windows.

"""


@tool
def get_system_info() -> tuple[str]:
    """
    Get information about the user's system like OS, processor, etc.

    Returns:
        A tuple containing system information.
    """

    info = platform.uname()
    return tuple(info)  # pyright: ignore[reportReturnType]


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
        name=take_screenshot.name,
        tool_call_id=tool_call_id,
    )


@tool
def click_screen(
    position: tuple[int, int],
    button: Button = Button.left,
    count: int = 1,
):
    """
    Clicks on the user's screen at the specified screen coordinates.

    Args:
        position: `(x, y)` screen coordinates to click at.
        button: Which mouse button to click with. Default is `Button.left`.
        count: How many times to click. Default is `1`.
    """

    controller = MouseController()
    controller.position = position
    time.sleep(0.5)
    controller.click(button, count)


@tool
def type_keyboard(text: str):
    """
    Type out a string of characters on the user's keyboard. Use this to type
    long pieces of text.

    Args:
        text: The text to type
    """

    controller = KeyboardController()
    controller.type(text)


@tool
def press_key(keys: list[str | ModifierKey]):
    """
    Press one or more keys on the user's keyboard at the same time.

    Args:
        keys:
            A list of keys to press. These can be either alphanumeric
            characters or `ModifierKey`s. DO NOT attempt to pass in modifier
            keys as plain strings.
    """

    mapped_keys = [
        MODIFIER_KEY_MAP[ModifierKey(k)] if len(str(k)) > 1 else str(k)
        for k in keys
    ]
    controller = KeyboardController()

    for key in mapped_keys:
        controller.press(key)
    for key in mapped_keys:
        controller.release(key)


def main():
    if not load_dotenv():
        print("Oops... Couldn't load .env file.")

    memory = InMemorySaver()
    agent = create_agent(
        model="anthropic:claude-haiku-4-5",
        system_prompt=SYSTEM_PROMPT,
        tools=[
            get_system_info,
            take_screenshot,
            click_screen,
            type_keyboard,
            press_key,
        ],
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

                if (
                    chunk_msg.type == "tool"  # pyright: ignore[reportAttributeAccessIssue]
                    and chunk_msg.name == take_screenshot.name  # pyright: ignore[reportAttributeAccessIssue]
                ):
                    chunk_msg = copy(chunk_msg)
                    chunk_msg.content = "[Output Hidden]"  # pyright: ignore[reportAttributeAccessIssue]

                print()
                chunk_msg.pretty_print()

        except KeyboardInterrupt:
            break


if __name__ == "__main__":
    main()
