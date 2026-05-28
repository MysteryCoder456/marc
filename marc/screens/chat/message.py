from typing import final, override

from langchain_core.messages import AnyMessage
from textual.app import RenderResult
from textual.widgets import Static


@final
class ChatMessage(Static):
    def __init__(self, msg: AnyMessage) -> None:
        super().__init__(id=f"msg-{msg.id}", classes="message", markup=False)

        self.msg = msg
        self.border_title = msg.type.capitalize()

    @override
    def render(self) -> RenderResult:
        return str(self.msg.content)
