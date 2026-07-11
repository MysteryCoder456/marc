from typing import Literal

from pydantic import BaseModel, Field

from marc.agent.tasks import Task


class ReasoningMessage(BaseModel):
    msg_type: Literal["reasoning"] = "reasoning"
    content: str


class TurnFinishedMessage(BaseModel):
    msg_type: Literal["turn_finished"] = "turn_finished"


class TasksUpdatedMessage(BaseModel):
    msg_type: Literal["tasks_updated"] = "tasks_updated"
    new_tasks: list[Task]


# Union of all possible message types.
# NOTE: Must be updated everytime a new message type is added.
OverlayMessageType = (
    ReasoningMessage | TurnFinishedMessage | TasksUpdatedMessage
)


class OverlayMessage(BaseModel):
    msg: OverlayMessageType = Field(discriminator="msg_type")
