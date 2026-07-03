# pyright: reportGeneralTypeIssues=false

import threading
import time
from pathlib import Path
from queue import Empty, Queue
from sys import stdin
from typing import final

import dearpygui.dearpygui as dpg
from mss import MSS

WHITE: tuple[int, int, int] = (255, 255, 255)
MUTED: tuple[int, int, int] = (120, 120, 120)


@final
class WorkOverlay:
    FONT_PATH = Path(__file__).parent / "MonaspiceKrNerdFont-Regular.otf"
    VIEWPORT_SIZE = (400, 150)
    FRAMES = ""
    FRAME_DURATION = 0.1
    TEXT_WRAP_MARGIN = 32

    def __init__(self):
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
        )

        self._build_ui()

        dpg.setup_dearpygui()
        dpg.show_viewport()
        dpg.set_primary_window("Work Mode", True)

        # Setup receiver thread
        self.recv_thread = threading.Thread(
            target=self._recv_loop, daemon=True
        )
        self.recv_thread.start()
        self.messages: Queue[str] = Queue()

    def __del__(self):
        dpg.destroy_context()

    def _build_ui(self):
        with dpg.item_handler_registry(tag="window handler"):
            dpg.add_item_resize_handler(callback=self._on_window_resize)

        with dpg.font_registry():
            dpg.add_font(str(self.FONT_PATH), 16, tag="font_md")
            dpg.add_font(str(self.FONT_PATH), 72, tag="font_face")

            # Global default font
            dpg.bind_font("font_md")

            # Enable nerd fonts
            dpg.add_font_range(0xE000, 0xF8FF)

        with dpg.window(tag="Work Mode") as window:
            with dpg.group():
                dpg.add_text("=D", tag="face")
                dpg.bind_item_font("face", "font_face")

                with dpg.group(horizontal=True):
                    dpg.add_text(self.FRAMES[0], color=MUTED, tag="indicator")
                    dpg.add_text(
                        "Ready",
                        color=MUTED,
                        tag="reasoning",
                        wrap=self.VIEWPORT_SIZE[0] - self.TEXT_WRAP_MARGIN,
                    )

                    dpg.hide_item("indicator")

            dpg.bind_item_handler_registry(window, "window handler")

    def _on_window_resize(self, _sender: str | int, _app_data: str | int):
        width = dpg.get_viewport_width()
        dpg.configure_item("reasoning", wrap=width - self.TEXT_WRAP_MARGIN)

    def _send_msg(self, data: str):
        """
        Send data to parent process via piped STDOUT.

        Args:
            data: Data to send as a string.
        """

        print(data, flush=True)

    def _recv_loop(self):
        """
        Receive data from parent process via piped STDIN and queue it for
        processing.
        """

        for line in stdin:
            self.messages.put(str(line))

    def render_loop(self):
        prev_now = time.time()

        while dpg.is_dearpygui_running():
            now = time.time()
            dt = now - prev_now

            # Process incoming messages
            while True:
                try:
                    msg = self.messages.get_nowait()
                except Empty:
                    break

                if msg.startswith("[reasoning]"):
                    reasoning = msg[11:].strip()
                    dpg.set_value("reasoning", reasoning)
                    dpg.show_item("indicator")

                elif msg.startswith("[done]"):
                    dpg.set_value("reasoning", "Done!")
                    dpg.hide_item("indicator")

            self.frame_elapsed += dt
            if self.frame_elapsed > self.FRAME_DURATION:
                self.frame_elapsed -= self.FRAME_DURATION
                self.indicator_frame = (self.indicator_frame + 1) % len(
                    self.FRAMES
                )
                dpg.set_value("indicator", self.FRAMES[self.indicator_frame])

            dpg.render_dearpygui_frame()
            prev_now = now
