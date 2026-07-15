from typing import final, override
from uuid import UUID, uuid4

from aenum import StrEnum
from pydantic import Field
from pydantic.dataclasses import dataclass


@final
class TaskStatus(StrEnum):
    TODO = "todo"
    IN_PROGRESS = "in_progress"
    COMPLETE = "complete"

    @property
    def icon(self) -> str:
        match self:
            case self.TODO:
                return "󰄰"
            case self.IN_PROGRESS:
                return "󱥸"
            case self.COMPLETE:
                return "󰄯"
            case _:
                return self


@dataclass
class Task:
    description: str = Field(max_length=50)

    id: UUID = Field(default_factory=uuid4)
    status: TaskStatus = TaskStatus.TODO  # pyright: ignore[reportAssignmentType]

    @override
    def __hash__(self) -> int:
        return hash((self.description, self.id, self.status))

    @override
    def __eq__(self, value: object, /) -> bool:
        if not isinstance(value, Task):
            return False

        return (
            self.id == value.id
            and self.description == value.description
            and self.status == value.status
        )
