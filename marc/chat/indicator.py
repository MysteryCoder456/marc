from typing import final, override

from textual.app import RenderResult
from textual.reactive import reactive
from textual.timer import Timer
from textual.widget import Widget


@final
class RunningIndicator(Widget):
    ANIMATION_FRAMES = [
        "···",
        "···",
        "•··",
        "·•·",
        "··•",
        "···",
        "···",
    ]
    FRAME_DURATION = 0.1

    current_frame_idx = reactive(0)

    def __init__(self) -> None:
        super().__init__(id="agent-running-indicator")

        self.timer: Timer = self.set_interval(
            self.FRAME_DURATION,
            self.update_frame,
            pause=True,
        )

    def show(self):
        self.styles.display = "block"
        self.current_frame_idx = 0
        self.timer.reset()  # also resumes the timer

    def hide(self):
        self.styles.display = "none"
        self.timer.pause()

    def update_frame(self):
        self.current_frame_idx = (self.current_frame_idx + 1) % len(
            self.ANIMATION_FRAMES
        )

    @override
    def render(self) -> RenderResult:
        current_frame = self.ANIMATION_FRAMES[self.current_frame_idx]
        return f"{current_frame} Working"
