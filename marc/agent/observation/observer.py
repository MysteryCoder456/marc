import asyncio
from dataclasses import dataclass
from typing import ClassVar, final

from textual import log
from textual.app import App
from textual.message import Message


@dataclass
class ObserverState:
    task: asyncio.Task
    app: App


class Observer:
    @final
    class Forward(Message):
        def __init__(self, content: str) -> None:
            super().__init__()
            self.content = content

    _state: ClassVar[ObserverState | None] = None

    @classmethod
    async def _screenshot_gate(cls) -> bool:
        # TODO: implement
        ...

    @classmethod
    async def _actionable_gate(cls) -> bool:
        # TODO: implement
        ...

    @classmethod
    async def _analyze_screenshot(cls, screenshot) -> str:
        # TODO: implement
        ...

    @classmethod
    async def _find_surfaceable_context(cls, context) -> list[str]:
        # TODO: implement
        ...

    @classmethod
    async def _task_loop(cls):
        # TODO: implement
        ...

    @classmethod
    def _forward_to_agent(cls, content: str):
        if not cls._state:
            raise ValueError("Observer state not set")

        cls._state.app.screen.post_message(cls.Forward(content))

    @classmethod
    def is_running(cls) -> bool:
        return cls._state is not None

    @classmethod
    def enter_observation_mode(cls, app: App):
        if cls._state:
            log(
                "Tried to enter observation mode while observer is already running"
            )
            return

        loop = asyncio.get_running_loop()
        task = loop.create_task(cls._task_loop())

        cls._state = ObserverState(task, app)

    @classmethod
    def exit_observation_mode(cls):
        if not cls._state:
            log("Tried to exit observation mode while observer is not running")
            return

        cls._state.task.cancel()
