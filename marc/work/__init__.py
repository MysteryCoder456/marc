# pyright: reportGeneralTypeIssues=false, reportUnnecessaryIsInstance=false

import threading
import time
from pathlib import Path
from queue import Empty, Queue
from sys import stdin
from typing import Any, final

import dearpygui.dearpygui as dpg
from mss import MSS

from .messages import (
    OverlayMessage,
    OverlayMessageType,
    ReasoningMessage,
    TasksUpdatedMessage,
    TurnFinishedMessage,
)

WHITE: tuple[int, int, int] = (255, 255, 255)
MUTED: tuple[int, int, int] = (120, 120, 120)

FONT_PATH = Path(__file__).parent / "MonaspiceKrNerdFont-Regular.otf"
VIEWPORT_SIZE = (400, 200)
TEXT_WRAP_MARGIN = 32

FACE_FRAMES = [
    ":\\",
    ":/",
]
FACE_FRAME_DURATION = 3.0

INDICATOR_FRAMES = ""
INDICATOR_FRAME_DURATION = 0.1


@final
class WorkOverlay:
    def __init__(self):
        self.face_frame = 0
        self.face_frame_elapsed = 0
        self.face_animating = False

        self.indicator_frame = 0
        self.indicator_frame_elapsed = 0
        self.indicator_animating = False

        self.task_ids: list[str] = []

        with MSS() as sct:
            mon = sct.monitors[1]
            viewport_position = (
                mon["width"] - VIEWPORT_SIZE[0],
                0,
            )
        dpg.create_context()
        dpg.create_viewport(
            title="Work Mode",
            x_pos=viewport_position[0],
            y_pos=viewport_position[1],
            width=VIEWPORT_SIZE[0],
            height=VIEWPORT_SIZE[1],
            always_on_top=True,
        )

        self._build_ui()

        dpg.setup_dearpygui()
        dpg.show_viewport()
        dpg.set_primary_window("Work Mode", True)

        # Setup receiver thread
        self.messages: Queue[OverlayMessageType] = Queue()
        self.recv_thread = threading.Thread(
            target=self._recv_loop, daemon=True
        )
        self.recv_thread.start()

    def __del__(self):
        dpg.destroy_context()

    def _build_ui(self):
        with dpg.item_handler_registry(tag="window handler"):
            dpg.add_item_resize_handler(callback=self._on_window_resize)

        with dpg.font_registry():
            dpg.add_font(str(FONT_PATH), 16, tag="font_md")
            dpg.add_font(str(FONT_PATH), 72, tag="font_face")

            # Global default font
            dpg.bind_font("font_md")

            # Enable nerd fonts
            dpg.add_font_range(0xE000, 0xF8FF)

        with dpg.window(tag="Work Mode") as window:
            with dpg.table(
                header_row=True,
                resizable=True,
                policy=dpg.mvTable_SizingStretchProp,
                tag="tasks_table",
            ):
                dpg.add_table_column(width_fixed=True)
                dpg.add_table_column(label="Tasks", width_stretch=True)

                with dpg.table_row():
                    with dpg.group(tag="face_group"):
                        dpg.add_text("=D", tag="face")
                        dpg.bind_item_font("face", "font_face")

                    dpg.add_group(tag="tasks")

            with dpg.group(horizontal=True):
                dpg.add_text("󰄬", color=MUTED, tag="indicator")
                dpg.add_text(
                    "Ready",
                    color=MUTED,
                    tag="reasoning",
                    wrap=VIEWPORT_SIZE[0] - TEXT_WRAP_MARGIN,
                )

            dpg.bind_item_handler_registry(window, "window handler")

    def _on_window_resize(self, _sender: str | int, _app_data: str | int):
        viewport_width = dpg.get_viewport_width()

        # Adjust reasoning text wrap
        dpg.configure_item("reasoning", wrap=viewport_width - TEXT_WRAP_MARGIN)

        # Adjust task text wrap
        face_width = dpg.get_item_rect_size("face_group")[0]
        task_wrap = viewport_width - face_width - TEXT_WRAP_MARGIN - 8
        for task_id in self.task_ids:
            dpg.configure_item(f"task-description-{task_id}", wrap=task_wrap)

    def _send_msg(self, data: Any):  # pyright: ignore[reportExplicitAny]
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
            serialized = str(line)
            msg = OverlayMessage.model_validate_json(serialized)
            self.messages.put(msg.msg)

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

                if isinstance(msg, ReasoningMessage):
                    dpg.set_value("reasoning", msg.content)

                    if not self.face_animating:
                        dpg.set_value("face", FACE_FRAMES[-1])
                    self.face_animating = True

                    if not self.indicator_animating:
                        dpg.set_value("indicator", INDICATOR_FRAMES[-1])
                    self.indicator_animating = True

                elif isinstance(msg, TurnFinishedMessage):
                    dpg.set_value("reasoning", "Done")

                    dpg.set_value("face", "=D")
                    self.face_animating = False

                    dpg.set_value("indicator", "󰄬")
                    self.indicator_animating = False

                elif isinstance(msg, TasksUpdatedMessage):
                    # Save new task ids
                    self.task_ids = [str(t.id) for t in msg.new_tasks]

                    # Remove existing tasks
                    dpg.delete_item("tasks", children_only=True)

                    # Add updated tasks
                    viewport_width = dpg.get_viewport_width()
                    face_width = dpg.get_item_rect_size("face_group")[0]
                    task_wrap = (
                        viewport_width - face_width - TEXT_WRAP_MARGIN - 8
                    )
                    for task in msg.new_tasks:
                        with dpg.group(horizontal=True, parent="tasks"):
                            dpg.add_text(
                                task.status.icon,
                                color=MUTED,
                                tag=f"task-indicator-{task.id}",
                            )
                            dpg.add_text(
                                task.description,
                                color=MUTED,
                                tag=f"task-description-{task.id}",
                                wrap=task_wrap,
                            )

            if self.face_animating:
                self.face_frame_elapsed += dt
                if self.face_frame_elapsed > FACE_FRAME_DURATION:
                    self.face_frame_elapsed -= FACE_FRAME_DURATION
                    self.face_frame = (self.face_frame + 1) % len(FACE_FRAMES)
                    dpg.set_value("face", FACE_FRAMES[self.face_frame])

            if self.indicator_animating:
                self.indicator_frame_elapsed += dt
                if self.indicator_frame_elapsed > INDICATOR_FRAME_DURATION:
                    self.indicator_frame_elapsed -= INDICATOR_FRAME_DURATION
                    self.indicator_frame = (self.indicator_frame + 1) % len(
                        INDICATOR_FRAMES
                    )
                    dpg.set_value(
                        "indicator", INDICATOR_FRAMES[self.indicator_frame]
                    )

            dpg.render_dearpygui_frame()
            prev_now = now
