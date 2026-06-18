import io
from asyncio import sleep
from base64 import b64encode
from dataclasses import dataclass
from typing import Annotated, Literal

from langchain.agents import create_agent
from langchain.tools import InjectedToolCallId, ToolRuntime, tool
from langchain_core.messages import ToolMessage
from langchain_core.messages.content import create_image_block
from langchain_openai import ChatOpenAI
from mss import MSS
from PIL import Image
from pynput.keyboard import Controller as KeyboardController
from pynput.mouse import Button
from pynput.mouse import Controller as MouseController

from .modifier_key import MODIFIER_KEY_MAP, ModifierKey

# Screenshots are downscaled to this width (preserving aspect ratio) before
# being sent to the model, so the model works in a smaller coordinate space.
TARGET_WIDTH = 1280


@dataclass
class ComputerContext:
    # Native screenshot width / TARGET_WIDTH for the most recent screenshot.
    # take_screenshot sets it; move_mouse uses it to map the model's
    # (downscaled-image-space) coordinates back to native screen pixels.
    scale_factor: float = 1.0


SYSTEM_PROMPT = """# Computer-Use Subagent System Prompt

You are a focused computer-use subagent. The main agent delegates visual
desktop tasks to you. Operate the user's screen carefully and report the
verified outcome with as few tool calls as possible.

## Cost Model

- HARD RULE: issue exactly ONE tool call per turn. Tool calls made in the
  same turn execute concurrently, which scrambles the order of mouse and
  keyboard actions.
- The whole conversation, including every screenshot, is re-sent on every
  turn. Screenshots are by far the most expensive item: every avoided action
  saves a turn, and every avoided screenshot saves the largest item there is.
- Budget: simple tasks should need 2-3 screenshots, most tasks at most 5.
  Past that, finish or report the blocker; do not keep polling.

## Strategy: Fewest Actions First

- Before touching the mouse, prefer the shortest action path: app launcher
  (cmd+space, type name, enter), browser address bar (cmd+L, type URL,
  enter), in-app search (cmd+F), menu shortcuts, tab/arrow navigation, enter
  to submit, esc to dismiss.
- Think silently; do not narrate plans or intermediate observations.
- Start with one screenshot. Reuse it for every target it still proves; take
  a new one only when an action changed the UI in a way you must see, before
  a risky confirmation, or for final evidence.
- Run an obvious low-risk sequence from one stable screenshot one action per
  turn without re-shooting between steps: focus field, type, press enter is
  three turns and zero extra screenshots.
- If the first screenshot already proves the goal is done, report completion
  immediately.

## Waiting and Verification

- After an action that triggers loading, animation, or focus changes, `wait`
  once (0.2-1s for small UI updates; 1-3s for page loads, app launches,
  submits), then screenshot once. Never take screenshots to watch progress.
- If a spinner or progress state persists, use at most two wait-and-check
  cycles, then report blocked or still loading.
- Do not wait when no visible or expected background change is pending.
- Verify typed text only when a mistake would matter or before submitting.
- Use the latest screenshot as final evidence when it already proves the
  result.
- After two failed attempts at the same target, stop and report the blocker.

## Mouse and Keyboard

- Estimate coordinates from screenshot landmarks; click directly when the
  target is clear. For precise or risky targets, move first, then click.
- If a click lands visibly offset from its target, the screenshot is likely
  captured at a higher pixel density than the pointer coordinate space:
  divide your coordinates by 2 (the typical HiDPI factor) and retry once.
- Prefer single clicks. Double-click, right-click, and drag only when the UI
  clearly requires them.
- Ensure the intended field, app, or control is focused before typing.
- `type_keyboard` for text; `press_key` for one chord of keys pressed
  together. Pass single characters as themselves and special keys by exact
  name: cmd, ctrl, alt, shift, enter, esc, tab, space, backspace, delete,
  home, end, page_up, page_down, up, down, left, right, f1-f20. Example:
  ["cmd", "l"].
- Scroll with the keyboard (page_down, page_up, arrows, space) after
  focusing the scrollable area.

## Safety

- Do not perform destructive, financial, account-changing, publishing,
  sending, purchasing, installing, permission-granting, or privacy-sensitive
  actions unless the delegated instruction explicitly asks for that action
  and the visible UI matches the instruction.
- Stop and report back if the UI asks for a password, MFA code, payment
  details, personal data, legal consent, or confirmation of an irreversible
  action.
- Do not guess hidden state. If a required target is not visible, search
  visually or scroll, else report the blocker.
- If an action has an unexpected result, recover only when the next safe
  step is clear from the latest screenshot.

## Final Report

At most two short sentences, addressed to the main agent:

- what was completed and the visible evidence; or
- the blocker, the last visible state, and any uncertainty.
"""


@tool
def take_screenshot(
    tool_call_id: Annotated[str, InjectedToolCallId],
    runtime: ToolRuntime[ComputerContext],
) -> ToolMessage:
    """
    Takes a screenshot of the user's desktop and returns it as a base64
    encoded image.
    """

    with MSS() as sct:
        # Grab screenshot
        monitor = sct.monitors[1]
        sct_img = sct.grab(monitor)

    img = Image.frombytes("RGB", sct_img.size, sct_img.rgb)

    # Downscale to TARGET_WIDTH (never upscale) so the model receives a
    # smaller image, and record how much it was shrunk so move_mouse can map
    # the model's coordinates back to native screen pixels.
    original_width, original_height = img.size
    if original_width > TARGET_WIDTH:
        target_height = round(original_height * TARGET_WIDTH / original_width)
        img = img.resize((TARGET_WIDTH, target_height), Image.LANCZOS)  # pyright: ignore[reportAttributeAccessIssue]
        runtime.context.scale_factor = original_width / TARGET_WIDTH
    else:
        runtime.context.scale_factor = 1.0

    # Save image data into a buffer
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
    runtime: ToolRuntime[ComputerContext],
):
    """
    Moves the cursor to the specified screen coordinates on the user's screen.
    Coordinates are in the pixel space of the latest screenshot (which is
    downscaled before being shown to you); they are scaled back up to the
    native screen resolution before being applied.

    Args:
        x: X-coordinate to move the mouse to.
        y: Y-coordinate to move the mouse to.
    """

    factor = runtime.context.scale_factor
    sx = round(x * factor)
    sy = round(y * factor)

    controller = MouseController()
    controller.position = (sx, sy)


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
    Press one chord of keys on the user's keyboard: all keys are pressed
    together, then released together.

    Args:
        keys:
            Keys in the chord. Pass single characters as themselves ("a",
            "1") and special keys by exact name: cmd, ctrl, alt, shift,
            enter, esc, tab, space, backspace, delete, home, end, page_up,
            page_down, up, down, left, right, caps_lock, f1-f20. Example:
            ["cmd", "shift", "t"].
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
        model="gpt-5.4-mini",
        use_responses_api=True,
        reasoning={"effort": "none"},
    )
    agent = create_agent(
        model=model,
        system_prompt=SYSTEM_PROMPT,
        context_schema=ComputerContext,
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
