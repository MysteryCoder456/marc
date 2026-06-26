import dearpygui.dearpygui as dpg


def face_changer(sender: int | str):
    match sender:
        case "foo":
            dpg.set_value("face", ":-D")
        case "bar":
            dpg.set_value("face", ":-(")
        case _:
            pass


def main():
    dpg.create_context()
    dpg.create_viewport(
        title="Work Mode",
        width=400,
        height=300,
        # decorated=False,
        always_on_top=True,
    )

    with dpg.window(tag="Work Mode"):  # pyright: ignore[reportGeneralTypeIssues]
        with dpg.draw_layer(tag="face"):  # pyright: ignore[reportGeneralTypeIssues]
            dpg.draw_line((10, 10), (60, 10), thickness=3)
            dpg.draw_line((10, 60), (60, 60), thickness=3)
            dpg.draw_bezier_quadratic(
                (110, 0), (160, 35), (110, 70), thickness=3
            )

    dpg.setup_dearpygui()
    dpg.show_viewport()
    dpg.set_primary_window("Work Mode", True)
    dpg.start_dearpygui()
    dpg.destroy_context()
