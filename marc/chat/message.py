from typing import final, override

from langchain_core.messages import AnyMessage, ToolCall, ToolMessage
from textual.app import ComposeResult
from textual.containers import VerticalGroup
from textual.reactive import reactive
from textual.widget import Widget
from textual.widgets import Markdown, Pretty, Static, Tree


@final
class ToolCallBlock(Widget):
    tool_result: reactive[ToolMessage | None] = reactive(None, recompose=True)

    def __init__(self, tool_call: ToolCall):
        super().__init__(
            id=f"tool-call-{tool_call['id']}", classes="tool-call"
        )
        self.tool_call = tool_call

    @property
    def tool_call_repr(self) -> str:
        args_str = ", ".join(
            f"{k}={v}" for k, v in self.tool_call["args"].items()
        )
        return f"{self.tool_call['name']}({args_str})"

    @property
    def tool_result_repr(self) -> list[str]:
        if not self.tool_result:
            return ["[dim i]Waiting for result...[/]"]

        output: list[str] = []
        for block in self.tool_result.content_blocks:
            block_repr: str
            if block["type"] == "text":
                block_repr = block["text"]
            else:
                block_repr = f"[dim][{block['type'].capitalize()} output][/]"
            output.append(block_repr)
        return output

    @override
    def compose(self) -> ComposeResult:
        t = Tree(self.tool_call_repr)

        for block_repr in self.tool_result_repr:
            t.root.add_leaf(block_repr)

        yield t


@final
class ChatMessageBlock(Static):
    TITLES = {
        "ai": "Marc",
    }

    def __init__(self, msg: AnyMessage) -> None:
        super().__init__(
            id=f"msg-{msg.id}",
            classes=f"message message-{msg.type}",
            markup=False,
        )

        self.content_blocks = msg.content_blocks

        if msg.type == "tool":
            self.border_title = f"Tool - [italic]{msg.name}[/]"
        else:
            self.border_title = self.TITLES.get(
                msg.type, msg.type.capitalize()
            )

    @override
    def compose(self) -> ComposeResult:
        with VerticalGroup():
            for block in self.content_blocks:
                match block["type"]:
                    case "text":
                        text = block["text"] or "No content"
                        yield Markdown(text)

                    case "reasoning":
                        reasoning = (
                            block.get("reasoning", "No reasoning")
                            or "No reasoning"
                        )
                        yield Markdown(reasoning, classes="reasoning-block")

                    case "tool_call":
                        pass

                    case "image":
                        pass

                    case _:
                        self.log(block)
                        yield Pretty(str(block))
