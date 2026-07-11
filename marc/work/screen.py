import asyncio
from asyncio import StreamWriter
from asyncio.subprocess import Process
from typing import final, override

from textual import work
from textual.app import ComposeResult
from textual.screen import Screen
from textual.widgets import Footer, Label

from .messages import OverlayMessage, OverlayMessageType


@final
class WorkModeScreen(Screen):
    CSS_PATH = "styles.tcss"
    BINDINGS = [
        ("ctrl+o", "disable_work_mode", "Disable Work Mode"),
    ]

    def __init__(self) -> None:
        super().__init__("work mode")

        self.overlay_process: Process | None = None

    def _get_process_stdin(self) -> StreamWriter | None:
        if proc := self.overlay_process:
            return proc.stdin

    async def send_message(self, msg: OverlayMessageType):
        if stdin := self._get_process_stdin():
            msg_outer = OverlayMessage(msg=msg)
            serialized = msg_outer.model_dump_json()
            stdin.write(f"{serialized}\n".encode())
            await stdin.drain()

    async def start_overlay(self):
        if self.overlay_process:
            self.log(
                "Warning: attempted to start overlay process when one already exists. Continuing with existing process."
            )
            return

        self.overlay_process = await asyncio.create_subprocess_exec(
            "python",
            "-m",
            "marc.work",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            stdin=asyncio.subprocess.PIPE,
        )
        self.await_overlay_close()

    async def close_overlay(self):
        if not self.overlay_process:
            return

        self.overlay_process.terminate()
        try:
            await asyncio.wait_for(self.overlay_process.wait(), timeout=3)
        except TimeoutError:
            self.overlay_process.kill()
            await self.overlay_process.wait()

    async def on_mount(self):
        # FIXME: on_mount isn't called when opening overlay for the second time.
        await self.start_overlay()

    async def on_unmount(self):
        await self.close_overlay()

    async def action_disable_work_mode(self):
        await self.close_overlay()

    @work
    async def await_overlay_close(self):
        """
        Waits for the overlay to close and automatically mutates the relevant
        internal state.
        """

        if not self.overlay_process:
            return

        await self.overlay_process.wait()
        self.overlay_process = None

        if self.is_running:
            self.app.pop_screen()

    @override
    def compose(self) -> ComposeResult:
        yield Label("Work Mode is Active 🐎")
        yield Footer(show_command_palette=False)
