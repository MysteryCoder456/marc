# pyright: reportGeneralTypeIssues=false

import time
from pathlib import Path
from queue import Queue
from typing import final

import dearpygui.dearpygui as dpg
from mss import MSS

WHITE: tuple[int, int, int] = (255, 255, 255)
MUTED: tuple[int, int, int] = (120, 120, 120)


@final
class WorkOverlay:
    FONT_PATH = Path(__file__).parent / "MonaspiceKrNerdFont-Regular.otf"
    VIEWPORT_SIZE = (300, 200)
    FRAMES = ""
    FRAME_DURATION = 0.1

    def __init__(self):
        self.to_dpg = Queue()
        self.from_dpg = Queue()

        self.indicator_frame = 0
        self.frame_elapsed = 0

        with MSS() as sct:
            mon = sct.monitors[1]
            viewport_position = (
                mon["width"] - self.VIEWPORT_SIZE[0],
                0,
            )
        dpg.create_context()
        dpg.create_viewport(
            title="Work Mode",
            x_pos=viewport_position[0],
            y_pos=viewport_position[1],
            width=self.VIEWPORT_SIZE[0],
            height=self.VIEWPORT_SIZE[1],
            always_on_top=True,
            # decorated=False,
        )

        # Register fonts
        with dpg.font_registry():
            dpg.add_font(str(self.FONT_PATH), 16, tag="font_md")
            dpg.add_font(str(self.FONT_PATH), 72, tag="font_face")

            # Global default font
            dpg.bind_font("font_md")

            # Enable nerd fonts
            dpg.add_font_range(0xE000, 0xF8FF)

        self._build_ui()

        dpg.setup_dearpygui()
        dpg.show_viewport()
        dpg.set_primary_window("Work Mode", True)

    def __del__(self):
        dpg.destroy_context()

    def _build_ui(self):
        with dpg.window(tag="Work Mode"):
            with dpg.group():
                dpg.add_text("=D", tag="face")
                dpg.bind_item_font("face", "font_face")

                with dpg.group(horizontal=True):
                    dpg.add_text("Discombobulating", color=MUTED)
                    dpg.add_text(self.FRAMES[0], color=MUTED, tag="indicator")

    def render_loop(self):
        prev_now = time.time()

        while dpg.is_dearpygui_running():
            now = time.time()
            dt = now - prev_now

            self.frame_elapsed += dt
            if self.frame_elapsed > self.FRAME_DURATION:
                self.frame_elapsed -= self.FRAME_DURATION
                self.indicator_frame = (self.indicator_frame + 1) % len(
                    self.FRAMES
                )
                dpg.set_value("indicator", self.FRAMES[self.indicator_frame])

            dpg.render_dearpygui_frame()
            prev_now = now
