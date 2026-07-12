import io
from asyncio import sleep
from base64 import b64encode
from dataclasses import dataclass
from typing import Annotated, Literal

from langchain.agents import create_agent
from langchain.tools import InjectedToolCallId, ToolRuntime, tool
from langchain_anthropic import ChatAnthropic
from langchain_anthropic.middleware import AnthropicPromptCachingMiddleware
from langchain_core.messages import ImageContentBlock, ToolMessage
from langchain_core.messages.content import create_image_block
from langchain_openai import ChatOpenAI
from mss import MSS, ScreenShot
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
    # take_screenshot sets it; click_mouse uses it to map the model's
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
- The user may have multiple screens, numbered from 1. `take_screenshot`
  captures one screen when given a `screen_number`, or every screen in a
  single call when given none — request all only when needed, never one
  call per screen.
  - Request all screens when screenshot context could be missing or stale:
    your first screenshot of the task, or right after an action that could
    open a new window or app, since either could land on any screen.
  - Otherwise default to the single screen you already know holds the
    target.
  - If you're unsure how many screens exist, request all.
- Reuse the latest screenshot for every target it still proves; take a new
  one only when an action changed the UI in a way you must see, before a
  risky confirmation, or for final evidence.
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

## Mouse, Keyboard, and Screens

- `click_mouse` moves the cursor to the given coordinates and clicks in one
  call — there is no separate move step, so estimate coordinates from
  screenshot landmarks and click directly in a single call.
- Pass coordinates exactly as seen in the screenshot you're reading; the
  tool rescales them to native screen pixels internally. Do not adjust them
  yourself (e.g. for HiDPI) — that double-scales and causes a miss.
- Pass the `screen_number` (1-indexed) of the screenshot the coordinates
  came from — if that screenshot covered every screen, identify which one
  shows the target first.
- If a click misses, take a fresh screenshot before retrying — don't reuse
  coordinates estimated from a stale image.
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
    screen_number: int | None = None,
) -> ToolMessage:
    """
    Takes screenshots of the user's desktops and returns them as base64
    encoded images. The user could have one or multiple screens.

    Args:
        screen_number:
            Which screen to take a screenshot of. First is 1, second is 2, etc.
            Pass `None` to get all screens.
    """

    # Grab screenshot(s)
    with MSS() as sct:
        if screen_number:
            monitors = [sct.monitors[screen_number]]
        else:
            monitors = sct.monitors[1:]

        sct_imgs: list[ScreenShot] = [
            sct.grab(monitor) for monitor in monitors
        ]

    def convert(sct_img: ScreenShot) -> ImageContentBlock:
        img = Image.frombytes("RGB", sct_img.size, sct_img.rgb)

        # Downscale to TARGET_WIDTH (never upscale) so the model receives a
        # smaller image, and record how much it was shrunk so click_mouse can map
        # the model's coordinates back to native screen pixels.
        original_width, original_height = img.size
        if original_width > TARGET_WIDTH:
            target_height = round(
                original_height * TARGET_WIDTH / original_width
            )
            img = img.resize((TARGET_WIDTH, target_height), Image.LANCZOS)  # pyright: ignore[reportAttributeAccessIssue]
            runtime.context.scale_factor = original_width / TARGET_WIDTH
        else:
            runtime.context.scale_factor = 1.0

        # Save image data into a buffer
        buf = io.BytesIO()
        img.save(buf, format="JPEG", optimize=True)
        img_bytes = buf.getvalue()

        return create_image_block(
            base64=b64encode(img_bytes).decode("utf-8"),
            mime_type="image/jpeg",
            detail="original",
        )

    return ToolMessage(
        content_blocks=[convert(img) for img in sct_imgs],
        name=take_screenshot.name,
        tool_call_id=tool_call_id,
    )


@tool
async def click_mouse(
    runtime: ToolRuntime[ComputerContext],
    screen_number: int,
    x: int,
    y: int,
    button: Literal["left", "middle", "right"] = "left",
    count: int = 1,
):
    """
    Moves the cursor to the specified screen coordinates on the user's screen
    and clicks. Coordinates are in the pixel space of the latest screenshot
    (which is downscaled before being shown to you); they are scaled back up
    to the native screen resolution before being applied.

    Args:
        screen_number: Which screen to click on. First is 1, second is 2, etc.
        x: X-coordinate to move the mouse to.
        y: Y-coordinate to move the mouse to.
        button: Which mouse button to click with. Default is `Button.left`.
        count: How many times to click. Default is `1`.
    """

    buttons = {
        "left": Button.left,
        "middle": Button.middle,
        "right": Button.right,
    }

    factor = runtime.context.scale_factor
    sx = round(x * factor)
    sy = round(y * factor)

    # Adjust for screen arrangement
    with MSS() as sct:
        monitor = sct.monitors[screen_number]
        sx += monitor["left"]
        sy += monitor["top"]

    controller = MouseController()
    controller.position = (sx, sy)
    await sleep(0.5)
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
        model="gpt-5.6-terra",
        use_responses_api=True,
        reasoning={"effort": "none"},
        temperature=0.5,
    )
    # model = ChatAnthropic(
    #     model="claude-haiku-4-5",  # pyright: ignore[reportCallIssue]
    #     effort="low",
    #     temperature=0.5,
    # )

    middleware = []
    if isinstance(model, ChatAnthropic):
        middleware.extend([AnthropicPromptCachingMiddleware()])

    agent = create_agent(
        model=model,
        system_prompt=SYSTEM_PROMPT,
        context_schema=ComputerContext,
        tools=[
            take_screenshot,
            click_mouse,
            type_keyboard,
            press_key,
            wait,
        ],
        middleware=middleware,
    )
    return agent
