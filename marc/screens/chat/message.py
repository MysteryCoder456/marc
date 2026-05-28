from typing import final, override

from langchain_core.messages import AnyMessage
from textual.app import ComposeResult
from textual.containers import VerticalGroup
from textual.widgets import Markdown, Pretty, Static


@final
class ChatMessage(Static):
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
        self.border_title = self.TITLES.get(msg.type, msg.type.capitalize())

    @override
    def compose(self) -> ComposeResult:
        with VerticalGroup():
            for block in self.content_blocks:
                match block["type"]:
                    case "text":
                        yield Markdown(block["text"])

                    case "tool_call":
                        yield Static(
                            f"[dim italic $text-accent]Tool Call: {block['name']}[/]"
                        )

                    case "image":
                        mime_type = block.get("mime_type", "unknown")
                        yield Static(f"[dim italic]Image: type={mime_type}[/]")

                    case "reasoning":
                        reasoning = block.get("reasoning", "No reasoning")
                        yield Static(f"[dim]{reasoning}[/]")

                    case _:
                        self.log(block)
                        yield Pretty(str(block))
