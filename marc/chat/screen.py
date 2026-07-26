import asyncio
from copy import deepcopy
from typing import final, override
from uuid import UUID

from langchain_core.messages import (
    AnyMessage,
    ToolMessage,
)
from langchain_core.runnables import Runnable
from textual import on, work
from textual.app import ComposeResult
from textual.binding import Binding
from textual.command import CommandPalette
from textual.containers import Horizontal, VerticalGroup, VerticalScroll
from textual.message import Message
from textual.reactive import reactive
from textual.screen import Screen
from textual.widget import Widget
from textual.widgets import Footer, Input, Tree
from textual.worker import Worker, WorkerState

from marc.agent import create_new_agent
from marc.agent.chat_name import generate_chat_name
from marc.agent.context import RuntimeContext, create_runtime_context
from marc.agent.memory import ShortTermMemory
from marc.work.messages import (
    ReasoningMessage,
    TasksUpdatedMessage,
    TurnFinishedMessage,
)
from marc.work.screen import WorkModeScreen

from .indicator import RunningIndicator
from .message import ChatMessageBlock, ToolCallBlock
from .providers import SkillListProvider
from .storage import ChatSession, ChatStorage


@final
class ContextPanel(Widget):
    # This reactive is a binding
    context = reactive(create_runtime_context, init=False, recompose=True)

    async def watch_context(self, context: RuntimeContext):
        wms = self.app.get_screen("work_mode", WorkModeScreen)
        await wms.send_message(
            TasksUpdatedMessage(new_tasks=context.current_tasks)
        )

    def _compose_tasks(self) -> ComposeResult:
        t = Tree("Tasks")
        t.root.expand_all()

        if self.context.current_tasks:
            for task in self.context.current_tasks:
                status_icon = task.status.icon
                t.root.add_leaf(f"{status_icon} {task.description}", task)
        else:
            t.root.add_leaf("No tasks yet")

        yield t

    @override
    def compose(self) -> ComposeResult:
        yield from self._compose_tasks()


@final
class ChatScreen(Screen):
    @final
    class Loaded(Message):
        def __init__(self, chat_id: UUID) -> None:
            super().__init__()
            self.chat_id = chat_id

    CSS_PATH = "styles.tcss"
    BINDINGS = [
        ("ctrl+o", "enable_work_mode", "Enable Work Mode"),
        ("ctrl+l", "toggle_context_panel", "Toggle Context Panel"),
        Binding("/", "open_skill_picker", show=False, priority=True),
    ]

    session: reactive[ChatSession] = reactive(ChatSession, init=False)
    session_context: reactive[RuntimeContext] = reactive(
        create_runtime_context, init=False
    )

    is_agent_running = reactive(False, init=False)
    is_showing_context_panel = reactive(False, init=False)

    def __init__(self, chat_id: UUID | None = None) -> None:
        super().__init__()

        self.chat_id = chat_id
        self.added_messages: set[str] = set()

        self.is_context_loaded = True
        self.is_context_panel_introduced = False

        self.agent: Runnable

    def scroll_to_end(self):
        scroller = self.query_one("#chat-scroll-area")
        scroller.scroll_end(animate=False)

    # ================ ↓ TEXTUAL FUNCTIONS ↓ ================

    async def watch_session(self, session: ChatSession):
        self.session_context = deepcopy(self.session.context)

        # Open context panel when tasks are added for the first time in this chat
        if (
            self.session_context.current_tasks
            and not self.is_context_panel_introduced
        ):
            self.is_showing_context_panel = True
            self.is_context_panel_introduced = True

        # Find newly added messages
        new_msgs = [
            msg
            for msg in session.messages
            if msg.id not in self.added_messages
        ]
        if not new_msgs:
            return
        self.added_messages.update([msg.id for msg in new_msgs])  # pyright: ignore[reportArgumentType]

        msg_container = self.query_one("#messages")
        tool_results: list[ToolMessage] = []

        # Mount new message widgets
        msg_widgets: list[Widget] = []
        for msg in new_msgs:
            if msg.type == "tool":
                tool_results.append(msg)
                continue

            if msg.type == "ai":
                # Only include chat message if non-tool calls exist
                if len(msg.content_blocks) > len(msg.tool_calls):
                    msg_widgets.append(ChatMessageBlock(msg))
                msg_widgets.extend(
                    [ToolCallBlock(tc) for tc in msg.tool_calls]
                )
            else:
                msg_widgets.append(ChatMessageBlock(msg))
        await msg_container.mount_all(msg_widgets)

        # Add tool call results under corresponding tool call block
        for msg in tool_results:
            tool_call = msg_container.query_one(
                f"#tool-call-{msg.tool_call_id}", ToolCallBlock
            )
            tool_call.tool_result = msg

        self.scroll_to_end()

    def watch_is_agent_running(self, running: bool):
        indicator = self.query_one(RunningIndicator)

        if running:
            indicator.show()
        else:
            indicator.hide()

    def watch_is_showing_context_panel(self, showing: bool):
        panel = self.query_one(ContextPanel)
        panel.styles.display = "block" if showing else "none"

    async def action_enable_work_mode(self):
        await self.app.push_screen("work_mode")

        # Show current tasks on overlay
        await asyncio.sleep(3)  # HACK: wait until overlay is actually visible
        wms = self.app.get_screen("work_mode", WorkModeScreen)
        await wms.send_message(
            TasksUpdatedMessage(new_tasks=self.session_context.current_tasks)
        )

    def action_toggle_context_panel(self):
        self.is_showing_context_panel = not self.is_showing_context_panel

    def action_open_skill_picker(self):
        self.app.push_screen(
            CommandPalette(
                [SkillListProvider], placeholder="Search for skills…"
            )
        )

    async def on_mount(self):
        # Initialize agent
        self.agent = await create_new_agent()

        if self.chat_id and (ses := await ChatStorage.load_chat(self.chat_id)):
            # Open existing chat
            self.session = ses
            self.is_context_loaded = False

        else:
            # Open new chat
            self.chat_id = self.session.id

        self.app.post_message(ChatScreen.Loaded(self.chat_id))

        # Focus input
        self.query_one("#chat-input").focus()

    async def on_unmount(self):
        # Don't do anything if conversation state hasn't changed
        if not (
            self.chat_id and self.session.messages and self.is_context_loaded
        ):
            return

        # Clean up
        save_coro = ChatStorage.save_chat(self.session)
        stm_coro = ShortTermMemory.save(self.session)
        self.run_worker(asyncio.gather(save_coro, stm_coro))

        # Wait for clean up if we're closing entire app
        if not self.app.is_running:
            await self.workers.wait_for_complete()

    @on(SkillListProvider.SkillSelected)
    def on_skill_selected(self, event: SkillListProvider.SkillSelected):
        chat_input = self.query_one("#chat-input", Input)
        chat_input.insert_text_at_cursor(f"/{event.skill_name}")

    @on(Input.Submitted, "#chat-input")
    def on_chat_input_submitted(self, event: Input.Submitted):
        msg = event.value
        event.control.validate(msg)
        if not event.control.is_valid:
            return

        # Reset input
        event.control.value = ""
        event.control.focus()

        self.send_message_to_agent(msg)

    @on(Worker.StateChanged)
    def on_agent_worker_state_changed(self, event: Worker.StateChanged):
        if event.worker.name != "agent_message":
            return

        match event.state:
            case WorkerState.PENDING:
                pass

            case WorkerState.RUNNING:
                self.is_agent_running = True

            case _:
                self.is_agent_running = False
                self.scroll_to_end()

    @work(name="agent_message")
    async def send_message_to_agent(self, msg: str):
        # Construct query
        query_messages: list[AnyMessage | dict[str, str]] = [
            {"role": "user", "content": msg}
        ]
        if not self.is_context_loaded:
            # Inject session's previous messages into context
            query_messages = self.session.messages + query_messages
            self.is_context_loaded = True

        context_hash = hash(self.session.context)
        wms = self.app.get_screen("work_mode", WorkModeScreen)

        try:
            # Send message to agent
            next_msg_idx = len(self.session.messages)
            response = self.agent.astream(
                {"messages": query_messages},
                {"configurable": {"thread_id": self.chat_id}},
                stream_mode="values",
                context=self.session.context,
            )

            # Stream agent responses
            async for chunk in response:
                # Update UI if context changed
                new_context_hash = hash(self.session.context)
                if new_context_hash != context_hash:
                    self.mutate_reactive(ChatScreen.session)
                    context_hash = new_context_hash

                chunk_msgs = chunk["messages"]
                new_msgs: list[AnyMessage] = chunk_msgs[next_msg_idx:]
                if not new_msgs:
                    continue

                # Send reasoning to overlay
                reasonings = []
                for block in new_msgs[-1].content_blocks:
                    if block["type"] != "reasoning":
                        continue

                    block_reasoning = (
                        str(block.get("reasoning")).replace("*", "").strip()
                    )
                    if block_reasoning:
                        reasonings.append(block_reasoning)
                if reasonings:
                    reasoning = ", ".join(reasonings).capitalize()
                    self.run_worker(
                        wms.send_message(ReasoningMessage(content=reasoning))
                    )

                self.session.messages.extend(new_msgs)
                self.mutate_reactive(ChatScreen.session)

                next_msg_idx = len(chunk_msgs)

        except Exception as e:
            self.log(e)
            raise

        # Signal to overlay that we're done
        self.run_worker(wms.send_message(TurnFinishedMessage()))

        # Generate a name for this session
        if not self.session.name:

            async def name_work():
                self.session.name = await generate_chat_name(
                    self.session.messages
                )

            self.run_worker(name_work())

    @override
    def compose(self) -> ComposeResult:
        with Horizontal():
            with VerticalScroll(id="chat-scroll-area"):
                yield VerticalGroup(id="messages")
                yield RunningIndicator()

            yield ContextPanel().data_bind(context=ChatScreen.session_context)

        with VerticalGroup(id="bottom-dock"):
            yield Input(placeholder="Chat", id="chat-input")
            yield Footer()
