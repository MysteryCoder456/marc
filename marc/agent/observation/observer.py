import asyncio
import time
from dataclasses import dataclass
from typing import ClassVar, final

from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from mss import MSS, ScreenShot
from PIL import Image, ImageChops
from pynput import keyboard, mouse
from textual import log
from textual.app import App
from textual.message import Message

from ..memory import LongTermMemory, ShortTermMemory, UserMemory
from ..utils import convert_to_img_block, search_long_term_memory


@dataclass
class ObserverState:
    app: App
    analysis_log: list[str]
    mouse_listener: mouse.Listener
    kb_listener: keyboard.Listener


class Observer:
    @final
    class Forward(Message):
        def __init__(self, content: str) -> None:
            super().__init__()
            self.content = content

    # maximum duration the user can stay idle for observer to stay active
    MAX_IDLE_DURATION: float = 10.0  # seconds

    _state: ClassVar[ObserverState | None] = None
    _task: ClassVar[asyncio.Task | None] = None
    _last_input_ts: ClassVar[float] = 0
    _prev_screens: ClassVar[list[ScreenShot]] = []

    @classmethod
    def is_running(cls) -> bool:
        return cls._state is not None

    @classmethod
    def _get_model(cls) -> ChatOpenAI:
        return ChatOpenAI(model="gpt-5.6-luna", reasoning={"effort": "low"})

    @classmethod
    def _forward_to_agent(cls, content: str):
        if not cls._state:
            raise ValueError("Observer state not set")

        cls._state.app.screen.post_message(cls.Forward(content))

    @classmethod
    def _forward_analysis(cls):
        if not cls._state:
            raise ValueError("Observer state not set")

        bulleted_log = "\n".join(
            [
                f"{i + 1}. {line}"
                for i, line in enumerate(cls._state.analysis_log)
            ]
        )
        final_forward = "# EXITING OBSERVATION MODE\n\n" + bulleted_log
        cls._forward_to_agent(final_forward)

    @classmethod
    async def _screenshot_gate(cls, screens: list[ScreenShot]) -> bool:
        try:
            # different number of screens? check them
            if len(screens) != len(cls._prev_screens):
                return True

            screen_zip = zip(screens, cls._prev_screens, strict=True)
            for screen, prev_screen in screen_zip:
                # cross-check dimensions
                if screen.size != prev_screen:
                    return True

                screen_pil = Image.frombytes("RGB", screen.size, screen.rgb)
                prev_screen_pil = Image.frombytes(
                    "RGB", screen.size, screen.rgb
                )
                diff = ImageChops.difference(screen_pil, prev_screen_pil)

                # check proportion of changed area
                def tf(x: int) -> float:
                    return int(x >= 50) * 255

                screen_area = diff.size[0] * diff.size[1]
                changed = diff.point(tf).histogram()[255]
                if changed / screen_area >= 0.05:
                    # more than 5% of area was changed
                    return True

            return False

        finally:
            cls._prev_screens = screens

    @classmethod
    async def _analyze_screenshots(cls, screens: list[ScreenShot]) -> str:
        prompt = (
            "You are the screen-analysis stage of an autonomous context "
            "surfacing mechanism for Marc, an AI desktop assistant. You will "
            "be given screenshots of the user's monitors, captured at one "
            "moment. This is your ONLY task.\n\n"
            "Describe what the user is doing right now, naming the specific "
            "applications, windows, files, and on-screen content you can "
            "see. If the screen supports it, add one sentence on what they "
            "appear to be trying to accomplish. Keep it under four "
            "sentences.\n\n"
            "Report only what is visible. Do not give advice or address the "
            "user; your output is read by another agent."
        )
        agent = create_agent(
            cls._get_model(),
            system_prompt=prompt,
        )

        blocks = [
            convert_to_img_block(
                Image.frombytes("RGB", screen.size, screen.rgb)
            )
            for screen in screens
        ]
        msg = HumanMessage(content_blocks=blocks)  # pyright: ignore[reportArgumentType]
        response = await agent.ainvoke({"messages": [msg]})
        return response["messages"][-1].content

    @classmethod
    async def _find_surfaceable_context(cls, analysis: str) -> str | None:
        user_memory, stm, ltm_index = await asyncio.gather(
            UserMemory.read(),
            ShortTermMemory.read(),
            LongTermMemory.read_index(),
        )

        prompt = (
            "You are the memory stage of an autonomous context surfacing "
            "mechanism for Marc, an AI desktop assistant. You will be given "
            "an analysis of what the user is doing right now, produced by "
            "the screen-analysis stage. Search Marc's memory for context "
            "that is useful and actionable to the user in this moment, and "
            "return it. This is your ONLY task.\n\n"
            "Start with the analysis itself. If it does not describe an "
            "activity specific enough to act on — an idle or empty screen, "
            "unrecognizable content, aimless browsing, a moment with no "
            "discernible goal — return `not found` immediately, without "
            "searching memory.\n\n"
            "Three tiers of memory are available to you:\n\n"
            "- User Profile: standing facts about the user — who they are, "
            "how they work, what they prefer. Included below in full. This "
            "is the only tier that holds facts about the user; use it to "
            "judge what would actually be useful to them, rarely as the "
            "surfaced context itself.\n"
            "- Short-Term Memory: a log of today's sessions, included below "
            "in full. Read it directly; there is nothing to search.\n"
            "- Long-Term Memory: facts from before today, held in a store "
            "you query with `search_long_term_memory`. The index below "
            "lists its topics grouped by category, telling you what exists "
            "there without the content. Use it to decide whether a search "
            "is worth running and to word the query — search a specific "
            "topic, project, or decision, never the raw analysis text. Two "
            "searches at most. LTM holds no facts about the user.\n\n"
            "Surface something only when it changes what the user would do "
            "next: a decision they are about to contradict, a detail they "
            "worked out before and would otherwise redo, an unresolved "
            "issue that this moment resolves. Return it in one or two "
            "sentences addressed to Marc, naming which memory it came from. "
            "Recognizing the user's activity is not itself a reason to "
            "surface anything.\n\n"
            "If memory holds nothing that meets that bar, return the phrase "
            "`not found` verbatim.\n\n"
            f"User Profile:\n{user_memory}\n\n"
            f"Short Term Memory:\n{stm}\n\n"
            f"Long Term Memory Index:\n{ltm_index}"
        )
        agent = create_agent(
            cls._get_model(),
            system_prompt=prompt,
            tools=[search_long_term_memory],
        )

        response = await agent.ainvoke(
            {"messages": [{"role": "human", "content": analysis}]}
        )
        response_text = response["messages"][-1].content

        if response_text == "not found":
            return None
        return response_text

    @classmethod
    async def _task_iter(cls):
        """
        Represents one complete traversal of the _observation flow_. See the
        Notion page titled `Observation`.
        """

        if not cls._state:
            raise ValueError("Observer state not set")

        # grab screenshot
        with MSS() as sct:
            monitors = sct.monitors[1:]
            scts: list[ScreenShot] = [
                sct.grab(monitor) for monitor in monitors
            ]

        if not await cls._screenshot_gate(scts):
            return

        # analyze grabbed screenshots
        analysis = await cls._analyze_screenshots(scts)
        cls._state.analysis_log.append(analysis)

        surfaceable = await cls._find_surfaceable_context(analysis)
        if not surfaceable:
            return

        # forward actionable context to main agent
        cls._forward_to_agent(surfaceable)

    @classmethod
    async def _task_loop(cls):
        try:
            while time.time() - cls._last_input_ts <= cls.MAX_IDLE_DURATION:
                iter_start = time.time()
                await cls._task_iter()
                iter_end = time.time()
                iter_duration = iter_end - iter_start

                # HACK: hardcoded iter rate to 1s
                await asyncio.sleep(1.0 - iter_duration)

        except asyncio.CancelledError:
            log("Observer task loop cancelled!")

        finally:
            cls._task = None

    @classmethod
    def _start_task(cls):
        cls._last_input_ts = time.time()

        if cls._task:
            return

        loop = asyncio.get_running_loop()
        cls._task = loop.create_task(cls._task_loop())

    @classmethod
    def _stop_task(cls):
        if cls._task:
            cls._task.cancel()
            cls._task = None

    @classmethod
    def enter_observation_mode(cls, app: App):
        if cls._state:
            log(
                "Tried to enter observation mode while observer is already running"
            )
            return

        # start input listeners
        mouse_listener = mouse.Listener(on_click=cls._start_task)
        mouse_listener.start()
        kb_listener = keyboard.Listener(on_press=cls._start_task)
        kb_listener.start()

        cls._state = ObserverState(app, [], mouse_listener, kb_listener)

    @classmethod
    def exit_observation_mode(cls):
        if not cls._state:
            log("Tried to exit observation mode while observer is not running")
            return

        # stop input listeners
        cls._state.mouse_listener.stop()
        cls._state.kb_listener.stop()

        # Stop the task loop if it's currently running
        cls._stop_task()

        # Forward accumulated analysis from this observation session
        cls._forward_analysis()

        cls._state = None
