# pyright: reportGeneralTypeIssues=false

import time
from pathlib import Path

import dearpygui.dearpygui as dpg
from mss import MSS

FONT_PATH = Path(__file__).parent / "MonaspiceKrNerdFont-Regular.otf"
VIEWPORT_SIZE = (300, 200)
WHITE: tuple[int, int, int] = (255, 255, 255)
MUTED: tuple[int, int, int] = (120, 120, 120)


def main():
    # Setup
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
        # decorated=False,
    )

    # Register fonts
    with dpg.font_registry():
        dpg.add_font(str(FONT_PATH), 16, tag="font_md")
        dpg.add_font(str(FONT_PATH), 72, tag="font_face")

        # Global default font
        dpg.bind_font("font_md")

        # Enable nerd fonts
        dpg.add_font_range(0xE000, 0xF8FF)

    indicator_frame = 0
    frame_elapsed = 0
    frame_duration = 0.1
    frames = ""

    # Build UI
    with dpg.window(tag="Work Mode"):
        with dpg.group():
            dpg.add_text("=D", tag="face")
            dpg.bind_item_font("face", "font_face")

            with dpg.group(horizontal=True):
                dpg.add_text("Discombobulating", color=MUTED)
                dpg.add_text(frames[0], color=MUTED, tag="indicator")

    # More setup
    dpg.setup_dearpygui()
    dpg.show_viewport()
    dpg.set_primary_window("Work Mode", True)

    # Render loop
    prev_now = time.time()
    while dpg.is_dearpygui_running():
        now = time.time()
        dt = now - prev_now

        frame_elapsed += dt
        if frame_elapsed > frame_duration:
            frame_elapsed -= frame_duration
            indicator_frame = (indicator_frame + 1) % len(frames)
            dpg.set_value("indicator", frames[indicator_frame])

        dpg.render_dearpygui_frame()
        prev_now = now

    # Cleanup
    dpg.destroy_context()
