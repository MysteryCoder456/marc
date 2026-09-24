import asyncio
import time
from concurrent.futures import Future
from dataclasses import dataclass
from typing import ClassVar, assert_never, final

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI
from mss import MSS, ScreenShot
from PIL import Image, ImageChops
from pynput import keyboard, mouse
from textual.app import App
from textual.message import Message

from ..memory import LongTermMemory, ShortTermMemory, UserMemory
from ..utils import convert_to_img_block, search_long_term_memory


@dataclass(frozen=True)
class AnalysisEvent:
    content: str


@dataclass(frozen=True)
class SurfacedContextEvent:
    content: str


type ObservationEvent = AnalysisEvent | SurfacedContextEvent


def _format_surfaceability_log(events: list[ObservationEvent]) -> str:
    lines = []
    for i, event in enumerate(events):
        if isinstance(event, AnalysisEvent):
            label = "ANALYSIS"
        elif isinstance(event, SurfacedContextEvent):
            label = "SURFACED CONTEXT"
        else:
            assert_never(event)
        lines.append(f"{i + 1}. {label}: {event.content}")
    return "\n".join(lines)


def _format_analysis_log(events: list[ObservationEvent]) -> str:
    analyses = (event for event in events if isinstance(event, AnalysisEvent))
    return "\n".join(
        f"{i + 1}. {event.content}" for i, event in enumerate(analyses)
    )


@dataclass
class ObserverState:
    app: App
    events: list[ObservationEvent]
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
    _task: ClassVar[Future | None] = None
    _last_input_ts: ClassVar[float] = 0
    _prev_screens: ClassVar[list[ScreenShot]] = []

    @classmethod
    def is_running(cls) -> bool:
        return cls._state is not None

    @classmethod
    def _get_model(cls) -> ChatOpenAI:
        return ChatOpenAI(model="gpt-6-luna", reasoning={"effort": "none"})

    @classmethod
    def _forward_to_agent(cls, content: str) -> bool:
        if not cls._state:
            raise ValueError("Observer state not set")

        return cls._state.app.screen.post_message(cls.Forward(content))

    @classmethod
    def _forward_analysis(cls):
        if not cls._state:
            raise ValueError("Observer state not set")

        bulleted_log = _format_analysis_log(cls._state.events)
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
                    return int(x >= 100) * 255

                screen_area = diff.size[0] * diff.size[1]
                changed = diff.point(tf).histogram()[255]
                if changed / screen_area >= 0.25:  # proportion of area changed
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
        response_msg: AIMessage = response["messages"][-1]
        response_text = "\n".join(
            [
                block.get("text") or ""
                for block in response_msg.content_blocks
                if block.get("text") is not None
            ]
        )
        return response_text

    @classmethod
    async def _find_surfaceable_context(cls, analysis: str) -> str | None:
        if not cls._state:
            raise ValueError("Observer state not set")

        surfaceability_log = _format_surfaceability_log(cls._state.events)
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
            "The current observation session's surfaceability transcript "
            "is included at the end. Earlier `ANALYSIS:` entries are "
            "supporting history that has already been evaluated. "
            "`SURFACED CONTEXT:` "
            "entries are the exact outputs already sent to Marc; do not "
            "surface them again or surface substantially repetitive "
            "context. The latest analysis is also provided separately as "
            "the human message and remains your primary input.\n\n"
            "If memory holds nothing that meets that bar, return the phrase "
            "`not found` verbatim.\n\n"
            f"User Profile:\n{user_memory}\n\n"
            f"Short Term Memory:\n{stm}\n\n"
            f"Long Term Memory Index:\n{ltm_index}\n\n"
            "Current Observation Session Transcript:\n"
            f"{surfaceability_log}"
        )
        agent = create_agent(
            cls._get_model(),
            system_prompt=prompt,
            tools=[search_long_term_memory],
        )

        response = await agent.ainvoke(
            {"messages": [{"role": "human", "content": analysis}]}
        )
        response_msg: AIMessage = response["messages"][-1]
        response_text = "\n".join(
            [
                block.get("text") or ""
                for block in response_msg.content_blocks
                if block.get("text") is not None
            ]
        )

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
        cls._state.events.append(AnalysisEvent(analysis))

        surfaceable = await cls._find_surfaceable_context(analysis)
        if not surfaceable:
            return

        # forward actionable context to main agent
        if cls._forward_to_agent(surfaceable):
            cls._state.events.append(SurfacedContextEvent(surfaceable))

    @classmethod
    async def _task_loop(cls):
        try:
            while time.time() - cls._last_input_ts <= cls.MAX_IDLE_DURATION:
                iter_start = time.time()
                await cls._task_iter()
                iter_end = time.time()
                iter_duration = iter_end - iter_start

                await asyncio.sleep(3.0 - iter_duration)

        except asyncio.CancelledError:
            print("Observer task loop cancelled!")

        finally:
            cls._task = None

    @classmethod
    def _start_task(cls):
        cls._last_input_ts = time.time()

        if cls._task or not cls._state:
            return

        loop = cls._state.app._loop  # pyright: ignore[reportPrivateUsage]
        if not loop:
            print("App event loop doesn't exist. This should not happen.")
            return
        cls._task = asyncio.run_coroutine_threadsafe(cls._task_loop(), loop)

    @classmethod
    def _stop_task(cls):
        if cls._task:
            cls._task.cancel()
            cls._task = None

    @classmethod
    def enter_observation_mode(cls, app: App):
        if cls._state:
            print(
                "Tried to enter observation mode while observer is already running"
            )
            return

        # start input listeners
        mouse_listener = mouse.Listener(on_click=cls._start_task)
        mouse_listener.start()
        kb_listener = keyboard.Listener(on_press=cls._start_task)
        kb_listener.start()

        cls._state = ObserverState(
            app=app,
            events=[],
            mouse_listener=mouse_listener,
            kb_listener=kb_listener,
        )

    @classmethod
    def exit_observation_mode(cls):
        if not cls._state:
            print(
                "Tried to exit observation mode while observer is not running"
            )
            return

        # stop input listeners
        cls._state.mouse_listener.stop()
        cls._state.kb_listener.stop()

        # Stop the task loop if it's currently running
        cls._stop_task()

        # Forward accumulated analysis from this observation session
        cls._forward_analysis()

        cls._state = None
