"""Canonical record schema for the judge's data.

ONE record shape serves three roles: human-labeled seed/test, frontier-teacher
distilled training labels, and the judge's own predictions at eval time. The
`label_source` field separates immovable human ground truth from machine labels.
"""

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from src.judge.task import AXES

_AXIS_NAMES = [a.name for a in AXES]
_MIN = {a.name: a.min_score for a in AXES}
_MAX = {a.name: a.max_score for a in AXES}
_SOURCES = {"human", "gpt-4o-mini"}


@dataclass(frozen=True)
class JudgeRecord:
    id: str
    question: str
    context: str  # what the answer must be grounded in
    answer: str  # the output being judged
    faithfulness: int  # label on the faithfulness axis
    relevance: int  # label on the relevance axis
    label_source: str  # "human" (gold) | "gpt-4o-mini" (distilled teacher)

    def __post_init__(self) -> None:
        for axis in _AXIS_NAMES:
            score = getattr(self, axis)
            if not _MIN[axis] <= score <= _MAX[axis]:
                raise ValueError(
                    f"{axis}={score} out of range "
                    f"[{_MIN[axis]}, {_MAX[axis]}] in record {self.id!r}"
                )
        if self.label_source not in _SOURCES:
            raise ValueError(f"unknown label_source {self.label_source!r}")


def load_records(path: Path) -> list[JudgeRecord]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [JudgeRecord(**json.loads(line)) for line in lines if line.strip()]


def save_records(records: list[JudgeRecord], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(asdict(r)) + "\n")
