import io
import os
import platform
import subprocess
from base64 import b64encode
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Annotated

from langchain.agents import create_agent
from langchain.tools import InjectedToolCallId, ToolRuntime, tool
from langchain_anthropic.middleware import AnthropicPromptCachingMiddleware
from langchain_core.messages import (
    ImageContentBlock,
    ToolMessage,
)
from langgraph.checkpoint.memory import InMemorySaver
from mss import MSS
from PIL import Image
from pynput.keyboard import Controller as KeyboardController
from pynput.keyboard import Key
from pynput.mouse import Button
from pynput.mouse import Controller as MouseController


@dataclass
class RuntimeContext:
    cwd: Path


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

SYSTEM_PROMPT = """# System Prompt

You are a competent coworker/colleague. You must help the user achieve various
tasks on their computer using the tools at your disposal.

Respond to all queries in a concise manner. If you are unsure about how to
respond or do something, just say so; you aren't perfect.

## Computer-use Tools

Several of the tools you have access to allow you to perform actions on the
user's computer. You must be extremely careful when using these, as the
actions you perform may be irreversible.

### Keyboard

When you type something using the keyboard, always verify that the result
is what you expected it to be before proceeding.

Also ensure that the right window/area is in focus before you attempt to
use keyboard shortcuts.

### Mouse

When you try to click something on the screen, move the mouse to the location
but do not click immediately. First verify that the mouse is in the correct
location, adjust its position if it isn't. Finally, click once you are sure
the mouse is placed correctly.
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
    img = Image.frombytes("RGB", sct_img.size, sct_img.rgb)
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
def move_mouse(
    position: tuple[int, int],
    delta: bool = False,
):
    """
    Moves the cursor to the specified screen coordinates on the user's screen.
    If `delta` is `True`, position is treated as an offset to the mouse's
    current position.

    Args:
        position: `(x, y)` screen coordinates to click at.
        delta: Whether to treat `position` as a relative offset.
    """

    controller = MouseController()

    if delta:
        controller.move(*position)
    else:
        controller.position = position


@tool
def click_mouse(
    button: Button = Button.left,
    count: int = 1,
):
    """
    Clicks on the user's screen at the cursor's current position.

    Args:
        button: Which mouse button to click with. Default is `Button.left`.
        count: How many times to click. Default is `1`.
    """

    controller = MouseController()
    controller.click(button, count)


@tool
def scroll_mouse(dx: int = 0, dy: int = 0):
    """
    Scroll's with the specified deltas in the x and y directions.

    Args:
        dx: How many pixels to scroll horizontally.
        dy: How many pixels to scroll vertically.
    """

    controller = MouseController()
    controller.scroll(dx, dy)


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


@tool
def read_file(path: Path, runtime: ToolRuntime[RuntimeContext]) -> str:
    """
    Read the contents of the file at the specified path.

    Args:
        path:
            Path to the file to be read. By default, this is interpreted as a
            relative path to the current working directory. Use the
            appropriate root path prefix to specify an absolute path (`C:\\` on
            Windows, `/` on Unix-based systems).

    Returns:
        Contents of the file as a string
    """

    if not path.is_absolute():
        path = runtime.context.cwd / path

    with open(path, "r") as f:
        return f.read()


@tool
def write_file(
    path: Path, contents: str, runtime: ToolRuntime[RuntimeContext]
):
    """
    Write to the file at the specified path. This will overwrite any existing
    data in the file.

    Args:
        path:
            Path to the file to be written to. By default, this is interpreted
            as a relative path to the current working directory. Use the
            appropriate root path prefix to specify an absolute path (`C:\\` on
            Windows, `/` on Unix-based systems).
        contents:
            New contents to be written to the file.
    """

    if not path.is_absolute():
        path = runtime.context.cwd / path

    with open(path, "w") as f:
        _ = f.write(contents)


@tool
def list_dir(path: Path, runtime: ToolRuntime[RuntimeContext]) -> list[str]:
    """
    List the files in directory at the specified path.

    Args:
        path:
            Path to the directory. By default, this is interpreted as a
            relative path to the current working directory. Use the
            appropriate root path prefix to specify an absolute path (`C:\\` on
            Windows, `/` on Unix-based systems).

    Return:
        A list of file and directory names in the specified directory
    """

    if not path.is_absolute():
        path = runtime.context.cwd / path

    return os.listdir(path)


@tool
def shell_command(
    cmd: str, runtime: ToolRuntime[RuntimeContext]
) -> str | tuple[str, str]:
    """
    Execute a shell command on the user's system at the current working
    directory. A timeout of 30s exists for any command executed.

    Always run non-interactive versions of commands whenever possible.

    Args:
        cmd: The shell command to execute.

    Returns:
        Output of the shell command as a tuple containing `(stdout, stderr)`,
        or the string `TIMEOUT` if the command timed out.
    """

    # TODO: handle Windows case

    user_shell = os.getenv("SHELL") or "/bin/bash"

    try:
        result = subprocess.run(
            [user_shell, "-i", "-c", cmd],
            capture_output=True,
            text=True,
            timeout=30,
            cwd=runtime.context.cwd,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
        )
        return result.stdout, result.stderr

    except subprocess.TimeoutExpired:
        return "TIMEOUT"


# TODO: tool to change CWD


def create_new_agent():
    memory = InMemorySaver()
    agent = create_agent(
        model="anthropic:claude-sonnet-4-6",
        system_prompt=SYSTEM_PROMPT,
        tools=[
            get_system_info,
            take_screenshot,
            move_mouse,
            click_mouse,
            scroll_mouse,
            type_keyboard,
            press_key,
            read_file,
            write_file,
            list_dir,
            shell_command,
        ],
        checkpointer=memory,
        context_schema=RuntimeContext,
        middleware=[AnthropicPromptCachingMiddleware()],  # pyright: ignore[reportArgumentType]
    )
    return agent

    # while True:
    #     try:
    #         user_input = input(">>> ")
    #         print()
    #
    #         response = agent.stream(
    #             {"messages": [{"role": "user", "content": user_input}]},
    #             {"configurable": {"thread_id": "thread-1"}},
    #             stream_mode="values",
    #             context=RuntimeContext(cwd=Path.cwd()),
    #         )
    #         for chunk in response:
    #             msg: AnyMessage = chunk["messages"][-1]
    #             print("=" * 30, msg.type.center(15), "=" * 30, end="\n" * 2)
    #
    #             for block in msg.content_blocks:
    #                 match block["type"]:
    #                     case "text":
    #                         print(">", block["text"])
    #
    #                     case "tool_call":
    #                         print(f"Calling Tool `{block['name']}` with args:")
    #                         pprint(block["args"])
    #
    #                     case "image":
    #                         block_copy = block.copy()
    #                         block_copy["base64"] = "<Truncated>"
    #                         pprint(block_copy)
    #
    #                     case _:
    #                         pprint(block)
    #
    #             print()
    #
    #     except KeyboardInterrupt:
    #         break
    #
    #     except Exception as e:
    #         print("!" * 30, "ERROR".center(15), "!" * 30, end="\n" * 2)
    #         pprint(e)
    #         print()
