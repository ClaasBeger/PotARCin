from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, Iterator, List

from framework.tasks.base import ArcTask


TaskLoader = Callable[[], Iterator[ArcTask]]


@dataclass
class TaskSet:
    """Named collection of tasks, produced by a loader function."""

    name: str
    loader: TaskLoader


class TaskRegistry:
    """Registry for named task sets (e.g. original ARC, synthetic, re_arc)."""

    def __init__(self) -> None:
        self._sets: Dict[str, TaskSet] = {}

    def register(self, task_set: TaskSet) -> None:
        if task_set.name in self._sets:
            raise ValueError(f"Task set already registered: {task_set.name}")
        self._sets[task_set.name] = task_set

    def get(self, name: str) -> TaskSet:
        try:
            return self._sets[name]
        except KeyError as exc:
            raise KeyError(f"Unknown task set: {name}") from exc

    def names(self) -> List[str]:
        return sorted(self._sets.keys())

    def iter_tasks(self, name: str) -> Iterator[ArcTask]:
        task_set = self.get(name)
        return task_set.loader()


GLOBAL_TASK_REGISTRY = TaskRegistry()

