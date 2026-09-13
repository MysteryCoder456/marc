import asyncio
import time
from dataclasses import dataclass
from typing import ClassVar, final

from mss import MSS, ScreenShot
from pynput import keyboard, mouse
from textual import log
from textual.app import App
from textual.message import Message


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

    @classmethod
    def is_running(cls) -> bool:
        return cls._state is not None

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
        # TODO: implement
        ...

    @classmethod
    async def _actionable_gate(cls, context: str) -> bool:
        # TODO: implement
        ...

    @classmethod
    async def _analyze_screenshots(cls, screens: list[ScreenShot]) -> str:
        # TODO: implement
        ...

    @classmethod
    async def _find_surfaceable_context(cls, analysis: str) -> str:
        # TODO: implement
        ...

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

        if not await cls._actionable_gate(surfaceable):
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
