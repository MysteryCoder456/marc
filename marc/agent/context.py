from pathlib import Path
from typing import override

from pydantic.dataclasses import dataclass

from .tasks import Task


@dataclass
class RuntimeContext:
    cwd: Path
    current_tasks: list[Task]

    @override
    def __hash__(self) -> int:
        return hash((self.cwd, *self.current_tasks))

    @override
    def __eq__(self, value: object, /) -> bool:
        if not isinstance(value, RuntimeContext):
            return False

        return hash(self) == hash(value)


def create_runtime_context() -> RuntimeContext:
    return RuntimeContext(
        cwd=Path.cwd(),
        current_tasks=[],
    )
