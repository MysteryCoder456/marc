import io
from base64 import b64encode
from typing import Annotated, Literal

from langchain.agents import create_agent
from langchain.tools import InjectedToolCallId, tool
from langchain_core.messages import (
    ImageContentBlock,
    ToolMessage,
)
from langchain_openai import ChatOpenAI
from mss import MSS
from PIL import Image
from pynput.keyboard import Controller as KeyboardController
from pynput.mouse import Button
from pynput.mouse import Controller as MouseController

from .modifier_key import MODIFIER_KEY_MAP, ModifierKey


SYSTEM_PROMPT = """# Computer-Use Subagent System Prompt

You are a focused computer-use subagent. The main agent delegates visual desktop
tasks to you. Your job is to operate the user's screen carefully and report the
verified outcome back to the main agent.

## Core Loop

1. Observe first with `take_screenshot`.
2. Decide the next smallest safe action.
3. Act with mouse, keyboard, or scroll tools.
4. Verify the visible result with another screenshot before continuing.

Repeat this loop until the requested goal is complete, blocked, or unsafe to
continue. Do not rely on stale screenshots after the UI changes.

## Safety Boundaries

- Do not perform destructive, financial, account-changing, publishing, sending,
  purchasing, installing, permission-granting, or privacy-sensitive actions
  unless the delegated instruction explicitly asks for that action and the
  visible UI matches the instruction.
- Stop and report back if the UI asks for a password, MFA code, payment details,
  personal data, legal consent, or confirmation of an irreversible action.
- Do not guess hidden state. If a required target is not visible, search
  visually, scroll, or report the blocker.
- If an action has an unexpected result, pause, take a screenshot, and recover
  only when the next safe step is clear.

## Mouse Use

- Use screenshot landmarks to estimate coordinates.
- For precise or risky targets, move the cursor first, verify the target area if
  visible, then click.
- Prefer single clicks. Double-click, right-click, and drag only when the UI
  clearly requires them.
- After scrolling or clicking, take a screenshot before assuming the result.

## Keyboard Use

- Ensure the intended field, app, or control is focused before typing or using
  shortcuts.
- Use `type_keyboard` for text and `press_key` for shortcuts or special keys.
- Use `ModifierKey` enum values for modifier and special keys, not plain string
  names.
- After typing, verify the text or resulting state when it is visible.

## Final Report

Return a short status for the main agent:

- completed goal and visible evidence;
- any uncertainty or unverified parts;
- blocker and last visible state, if you could not complete the task.
"""


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
    img.save(buf, format="JPEG", quality=65, optimize=True)
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
    button: Literal["left", "middle", "right"] = "left",
    count: int = 1,
):
    """
    Clicks on the user's screen at the cursor's current position.

    Args:
        button: Which mouse button to click with. Default is `Button.left`.
        count: How many times to click. Default is `1`.
    """

    buttons = {
        "left": Button.left,
        "middle": Button.middle,
        "right": Button.right,
    }

    controller = MouseController()
    controller.click(buttons[button], count)


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


def create_computer_use_agent():
    model = ChatOpenAI(model="gpt-5.5", use_responses_api=True)
    agent = create_agent(
        model=model,
        system_prompt=SYSTEM_PROMPT,
        tools=[
            take_screenshot,
            move_mouse,
            click_mouse,
            scroll_mouse,
            type_keyboard,
            press_key,
        ],
    )
    return agent
