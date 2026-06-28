# pyright: reportGeneralTypeIssues=false

import time

import dearpygui.dearpygui as dpg
from mss import MSS

VIEWPORT_SIZE = (300, 200)
WHITE: tuple[int, int, int] = (255, 255, 255)
MUTED: tuple[int, int, int] = (100, 100, 100)


def draw_face():
    with dpg.drawlist(width=150, height=70):
        with dpg.draw_layer(tag="face"):
            dpg.draw_line((0, 10), (50, 10), thickness=3)
            dpg.draw_line((0, 60), (50, 60), thickness=3)
            dpg.draw_bezier_quadratic(
                (100, 0), (150, 35), (100, 70), thickness=3
            )


def main():
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
        # decorated=False,
        always_on_top=True,
    )

    with dpg.window(tag="Work Mode"):
        with dpg.group():
            draw_face()

            with dpg.group(horizontal=True):
                dpg.add_text("Discombobulating", color=MUTED)

                for i in range(3):
                    dpg.add_text("·", color=MUTED, tag=f"indicator-{i}")

    dpg.setup_dearpygui()
    dpg.show_viewport()
    dpg.set_primary_window("Work Mode", True)

    indicator_frame = 0
    frame_elapsed = 0

    prev_now = time.time()
    while dpg.is_dearpygui_running():
        now = time.time()
        dt = now - prev_now

        frame_elapsed += dt
        if frame_elapsed > 0.1:
            frame_elapsed = 0

            if indicator_frame >= 0:
                dpg.configure_item(f"indicator-{indicator_frame}", color=MUTED)

            indicator_frame += 1
            if indicator_frame >= 3:
                frame_elapsed = -4 * 0.1
                indicator_frame = -1
            else:
                dpg.configure_item(f"indicator-{indicator_frame}", color=WHITE)

        dpg.render_dearpygui_frame()
        prev_now = now

    dpg.destroy_context()
