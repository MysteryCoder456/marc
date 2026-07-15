import asyncio
from asyncio import StreamWriter
from asyncio.subprocess import Process
from typing import final, override

from textual import work
from textual.app import ComposeResult
from textual.screen import Screen
from textual.widgets import Footer, Label
from textual.worker import Worker

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
        self.recv_worker: Worker | None = None

    def _get_process_stdin(self) -> StreamWriter | None:
        if proc := self.overlay_process:
            return proc.stdin

    async def send_message(self, msg: OverlayMessageType):
        if stdin := self._get_process_stdin():
            msg_outer = OverlayMessage(msg=msg)
            serialized = msg_outer.model_dump_json()
            try:
                stdin.write(f"{serialized}\n".encode())
                await stdin.drain()
            except (BrokenPipeError, ConnectionResetError):
                self.log(
                    "Warning: Failed to send message to overlay because the "
                    "process is no longer accepting input."
                )

    async def start_overlay(self):
        import sys

        if self.overlay_process:
            self.log(
                "Warning: attempted to start overlay process when one already exists. Continuing with existing process."
            )
            return

        self.overlay_process = await asyncio.create_subprocess_exec(
            sys.executable,
            "-m",
            "marc.work",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            stdin=asyncio.subprocess.PIPE,
        )
        self.recv_worker = self.recv_messages()
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

    # Not using `on_mount` because it only runs once in a single app lifecycle
    async def on_screen_resume(self):
        await self.start_overlay()

    # Would use `on_screen_suspend` for parity, but this is only meant to run
    # when the entire app shuts down.
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

        if self.recv_worker:
            self.recv_worker.cancel()

        if stderr := self.overlay_process.stderr:

            async def log_stderr():
                self.log((await stderr.read()).decode())

            self.run_worker(log_stderr())

        self.overlay_process = None

        if self.is_running:
            self.app.pop_screen()

    @work
    async def recv_messages(self):
        if not self.overlay_process:
            return

        stdout = self.overlay_process.stdout
        if not stdout:
            return

        while line := await stdout.readline():
            self.log(line.decode())

    @override
    def compose(self) -> ComposeResult:
        yield Label("Work Mode is Active 🐎")
        yield Footer(show_command_palette=False)
