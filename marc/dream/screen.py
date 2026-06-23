from typing import final, override

from textual import on
from textual.app import ComposeResult, RenderResult
from textual.message import Message
from textual.reactive import reactive
from textual.screen import Screen
from textual.widget import Widget
from textual.widgets import Static
from textual.worker import Worker, WorkerState

from marc.agent.memory import LongTermMemory


@final
class ZZZ(Widget):
    FRAMES = [
        "Z  ",
        "ZZ ",
        "ZZZ",
        "ZZZ",
        "   ",
        "   ",
    ]

    current_frame = reactive(0)

    def on_mount(self):
        self.set_interval(0.4, self.update_current_frame)

    def update_current_frame(self):
        self.current_frame = (self.current_frame + 1) % len(self.FRAMES)

    @override
    def render(self) -> RenderResult:
        return self.FRAMES[self.current_frame]


@final
class DreamModeScreen(Screen):
    @final
    class FinishedDreaming(Message):
        pass

    CSS_PATH = "styles.tcss"

    def on_mount(self):
        self.run_worker(LongTermMemory.dream(), name="dreamer")

    @on(Worker.StateChanged)
    def on_dreamer_state_changed(self, event: Worker.StateChanged):
        if event.worker.name != "dreamer":
            return

        match event.state:
            case WorkerState.RUNNING:
                self.log("Entering dream mode 😴")

            case WorkerState.SUCCESS:
                self.log("Finished dreaming 🥱")
                self.post_message(DreamModeScreen.FinishedDreaming())

            case _:
                pass

    @override
    def compose(self) -> ComposeResult:
        yield Static("Marc is Dreaming", classes="grr")
        yield ZZZ(classes="grr")
