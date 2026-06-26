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
        dpg.add_text(":-|", tag="face")

        with dpg.group():  # pyright: ignore[reportGeneralTypeIssues]
            dpg.add_button(label="Foo", tag="foo", callback=face_changer)
            dpg.add_button(label="Bar", tag="bar", callback=face_changer)

    dpg.setup_dearpygui()
    dpg.show_viewport()
    dpg.set_primary_window("Work Mode", True)
    dpg.start_dearpygui()
    dpg.destroy_context()
