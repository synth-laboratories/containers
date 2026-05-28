from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Iterable

from .capabilities import TaskCatalog, TaskInfo
from .nouns import TaskDefinition, TaskInstance


class DatasetSplit(StrEnum):
    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"
    OTHER = "other"

    @classmethod
    def parse(cls, value: Any) -> "DatasetSplit":
        if isinstance(value, cls):
            return value
        text = str(value or "").strip().lower().replace("-", "_")
        aliases = {
            "train": cls.TRAIN,
            "validation": cls.VALIDATION,
            "val": cls.VALIDATION,
            "dev": cls.VALIDATION,
            "test": cls.TEST,
        }
        return aliases.get(text, cls.OTHER)


@dataclass(frozen=True, slots=True)
class TaskFilter:
    task_family: str | None = None
    split: DatasetSplit | str | None = None
    tags: tuple[str, ...] = ()
    metadata_equals: dict[str, Any] = field(default_factory=dict)
    limit: int | None = None

    def normalized_split(self) -> DatasetSplit | None:
        if self.split is None:
            return None
        return DatasetSplit.parse(self.split)


def task_definition_from_task_info(task_info: TaskInfo) -> TaskDefinition:
    return TaskDefinition(
        task_id=task_info.task.task_id,
        task_name=task_info.task.task_name,
        task_family=task_info.task.task_family,
        description=task_info.task.description,
        version=task_info.task.version,
        benchmark=task_info.task.benchmark,
        metadata={**dict(task_info.task.metadata), **dict(task_info.task_metadata)},
    )


class InMemoryTaskCatalog:
    """Mutable convenience wrapper around `TaskCatalog`.

    This is intentionally lightweight and in-memory only. The key contract is the
    data shape, so implementations can be swapped with a persistent backing store
    (for example SQLite) without changing call sites.
    """

    def __init__(
        self,
        *,
        catalog_id: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self._catalog_id = str(catalog_id)
        self._metadata = dict(metadata or {})
        self._tasks: dict[str, TaskDefinition] = {}
        self._instances: dict[str, TaskInstance] = {}

    @classmethod
    def from_task_info(cls, task_info: TaskInfo) -> "InMemoryTaskCatalog":
        catalog = cls(
            catalog_id=f"{task_info.task.task_family or task_info.task.task_id}:catalog",
            metadata={"source": "task_info_seed"},
        )
        catalog.add_task(task_definition_from_task_info(task_info))
        return catalog

    def add_task(self, task: TaskDefinition) -> None:
        self._tasks[str(task.task_id)] = task

    def add_instance(self, instance: TaskInstance) -> None:
        if instance.task_id not in self._tasks:
            raise KeyError(f"unknown task_id: {instance.task_id}")
        self._instances[str(instance.task_instance_id)] = instance

    def get_task(self, task_id: str) -> TaskDefinition | None:
        return self._tasks.get(str(task_id))

    def get_instance(self, task_instance_id: str) -> TaskInstance | None:
        return self._instances.get(str(task_instance_id))

    def list_tasks(self) -> list[TaskDefinition]:
        return list(self._tasks.values())

    def list_instances(self, *, task_id: str | None = None) -> list[TaskInstance]:
        if task_id is None:
            return list(self._instances.values())
        target = str(task_id)
        return [item for item in self._instances.values() if item.task_id == target]

    def query(self, *, split: str | None = None, tags: Iterable[str] | None = None, limit: int | None = None) -> list[TaskInstance]:
        required_tags = {str(tag) for tag in (tags or []) if str(tag)}
        rows = list(self._instances.values())
        if split is not None:
            split_value = DatasetSplit.parse(split).value
            rows = [item for item in rows if DatasetSplit.parse(item.split).value == split_value]
        if required_tags:
            rows = [item for item in rows if required_tags.issubset(set(item.tags))]
        if limit is not None:
            rows = rows[: max(0, int(limit))]
        return rows

    def filter_instances(self, task_filter: TaskFilter) -> list[TaskInstance]:
        results: list[TaskInstance] = []
        target_split = task_filter.normalized_split()
        target_tags = {str(tag) for tag in task_filter.tags if str(tag)}
        for instance in self._instances.values():
            task = self._tasks.get(instance.task_id)
            if task_filter.task_family and (task is None or task.task_family != task_filter.task_family):
                continue
            if target_split is not None and DatasetSplit.parse(instance.split) is not target_split:
                continue
            if target_tags and not target_tags.issubset(set(instance.tags)):
                continue
            if any(instance.metadata.get(key) != value for key, value in task_filter.metadata_equals.items()):
                continue
            results.append(instance)
        if task_filter.limit is not None:
            results = results[: max(0, int(task_filter.limit))]
        return results

    def to_task_catalog(self) -> TaskCatalog:
        return TaskCatalog(
            catalog_id=self._catalog_id,
            tasks=self.list_tasks(),
            instances=self.list_instances(),
            metadata=dict(self._metadata),
        )

    def to_dict(self) -> dict[str, Any]:
        return self.to_task_catalog().to_dict()
