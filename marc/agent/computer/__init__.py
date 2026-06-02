import io
from asyncio import sleep
from base64 import b64encode
from typing import Annotated, Literal

from langchain.agents import create_agent
from langchain.tools import InjectedToolCallId, tool
from langchain_core.messages import ToolMessage
from langchain_core.messages.content import create_image_block
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
verified outcome back to the main agent with minimal tool calls and minimal
text.

## Action Economy

- Think silently. Do not narrate intermediate observations or plans.
- Start with one screenshot. Reuse it until an action changes the UI, the target
  is uncertain, or verification is needed.
- Do not take a screenshot only to verify cursor position. If a visible target
  is clear and low-risk, move and click without an intermediate screenshot.
- Run short, obvious sequences from one stable screenshot when each step is
  low-risk, such as focusing a field, typing text, and pressing Enter.
- After a click, submit, scroll, shortcut, or app switch that may trigger
  loading, animation, or focus changes, use `wait` briefly before taking the
  next screenshot.
- Prefer keyboard shortcuts, search fields, and direct text entry when they are
  likely faster than visual navigation.
- Prefer keyboard scrolling over mouse scrolling when the scrollable area is
  focused or can be focused cheaply.
- If the first screenshot already proves the goal is done, report completion
  without more tools.
- After two failed attempts at the same target, stop and report the blocker.

## Verification

- Take a new screenshot after actions that likely changed the visible state,
  before risky confirmations, after unexpected results, and before final
  reporting if the current evidence is stale.
- For routine navigation or form entry, one screenshot after a safe action
  sequence is enough.
- Prefer one short `wait` plus one screenshot over repeated immediate
  screenshots while the UI is still settling.
- Use the latest screenshot as final evidence when it already proves the result.

## Waiting

- Use `wait(0.2)` to `wait(1)` for small UI updates, focus changes, menus,
  animations, and short transitions.
- Use `wait(1)` to `wait(3)` after page loads, app launches, submits, or other
  actions expected to take longer.
- If a spinner or progress state remains visible, use at most two wait-and-check
  cycles before reporting that the task is blocked or still loading.
- Do not wait when no visible or expected background change is pending.

## Safety

- Do not perform destructive, financial, account-changing, publishing, sending,
  purchasing, installing, permission-granting, or privacy-sensitive actions
  unless the delegated instruction explicitly asks for that action and the
  visible UI matches the instruction.
- Stop and report back if the UI asks for a password, MFA code, payment details,
  personal data, legal consent, or confirmation of an irreversible action.
- Do not guess hidden state. If a required target is not visible, search
  visually, scroll, or report the blocker.
- If an action has an unexpected result, recover only when the next safe step is
  clear from the latest screenshot.

## Mouse Use

- Use screenshot landmarks to estimate coordinates.
- Click directly when the target is clear. For precise or risky targets, move
  first and only re-check if needed.
- Prefer single clicks. Double-click, right-click, and drag only when the UI
  clearly requires them.

## Keyboard Use

- Ensure the intended field, app, or control is focused before typing or using
  shortcuts.
- Use `type_keyboard` for text and `press_key` for shortcuts or special keys.
- Use `ModifierKey` enum values for modifier and special keys, not plain string
  names.
- For scrolling, prefer `page_down`, `page_up`, `down`, `up`, `home`, `end`,
  or `space` with `press_key` after focusing the scrollable area.
- Verify typed text only when mistakes would matter or before submitting.

## Final Report

Return at most two short sentences:

- what was completed and the visible evidence; or
- the blocker, last visible state, and any uncertainty.

## **CRITICAL**

You are NOT allowed to call multiple tools in one turn. Only use one tool
at a time.
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
            create_image_block(
                base64=b64encode(img_bytes).decode("utf-8"),
                mime_type="image/jpeg",
                detail="original",
            ),
        ],
        name=take_screenshot.name,
        tool_call_id=tool_call_id,
    )


@tool
def move_mouse(
    x: int,
    y: int,
    delta: bool = False,
):
    """
    Moves the cursor to the specified screen coordinates on the user's screen.
    If `delta` is `True`, x and y are treated as offsets relative to the
    mouse's current position.

    Args:
        x: X-coordinate to move the mouse to.
        y: Y-coordinate to move the mouse to.
        delta: Whether to treat `position` as a relative offset.
    """

    controller = MouseController()

    if delta:
        controller.move(x, y)
    else:
        controller.position = (x, y)


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
async def wait(seconds: float):
    """
    Wait for a specified amount of time before continuing.

    Args:
        seconds: How many seconds to wait for.
    """

    await sleep(seconds)


def create_computer_use_agent():
    model = ChatOpenAI(
        model="gpt-5.4",
        use_responses_api=True,
        reasoning={"effort": "none"},
    )
    agent = create_agent(
        model=model,
        system_prompt=SYSTEM_PROMPT,
        tools=[
            take_screenshot,
            move_mouse,
            click_mouse,
            type_keyboard,
            press_key,
            wait,
        ],
    )
    return agent
