"""Label the unlabeled pool with the teacher (GPT-4o-mini) THROUGH judgekit's
LLMJudge -- the same library and code path that will benchmark the trained judge
later. This is the distillation: the teacher's scores become training labels.

Before splitting, we sanity-check the teacher's labels against the `intended`
tags. If the expected pattern doesn't hold, the teacher is mislabeling -- and we
fix that BEFORE training, not after.

Usage: uv run python -m scripts.label_pool
"""

import json
import random
from pathlib import Path

from judgekit import Axis, LLMJudge, Sample

from src.data.schema import JudgeRecord, save_records
from src.judge.task import AXES
from src.teacher import get_completer

POOL = Path("data/pool.jsonl")
TRAIN = Path("datasets/train.jsonl")
VAL = Path("datasets/val.jsonl")
VAL_FRACTION = 0.2
SEED = 0


def label() -> list[tuple[str, JudgeRecord]]:
    """Return (intended_tag, teacher-labeled record) pairs."""
    jk_axes = [Axis(a.name, a.description, a.min_score, a.max_score) for a in AXES]
    judge = LLMJudge(jk_axes, get_completer())
    rows = [
        json.loads(line)
        for line in POOL.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    labeled: list[tuple[str, JudgeRecord]] = []
    for i, row in enumerate(rows):
        try:
            judgment = judge.score(
                Sample(
                    input=row["question"],
                    output=row["answer"],
                    context=row["context"],
                )
            )
        except Exception as exc:  # one flaky judge call shouldn't kill the run
            print(f"  [skip {i}: {exc}]")
            continue
        labeled.append(
            (
                row["intended"],
                JudgeRecord(
                    id=f"distil_{i:04d}",
                    question=row["question"],
                    context=row["context"],
                    answer=row["answer"],
                    faithfulness=judgment.scores["faithfulness"],
                    relevance=judgment.scores["relevance"],
                    label_source="gpt-4o-mini",
                ),
            )
        )
        if (i + 1) % 20 == 0:
            print(f"  ...labeled {i + 1}")
    return labeled


def audit(labeled: list[tuple[str, JudgeRecord]]) -> None:
    print("\nSanity check -- mean teacher scores by intended flavor:")
    print(f"  {'intended':12}{'faithfulness':>14}{'relevance':>12}   n")
    for intent in ("good", "unfaithful", "off_topic"):
        recs = [r for tag, r in labeled if tag == intent]
        if not recs:
            continue
        mf = sum(r.faithfulness for r in recs) / len(recs)
        mr = sum(r.relevance for r in recs) / len(recs)
        print(f"  {intent:12}{mf:>14.2f}{mr:>12.2f}   {len(recs)}")
    print(
        "\nExpected: good -> both high; unfaithful -> faithfulness low; "
        "off_topic -> relevance low."
    )


def main() -> None:
    labeled = label()
    audit(labeled)
    records = [r for _, r in labeled]
    rng = random.Random(SEED)
    rng.shuffle(records)
    n_val = int(len(records) * VAL_FRACTION)
    save_records(records[n_val:], TRAIN)
    save_records(records[:n_val], VAL)
    print(f"\nWrote {len(records) - n_val} train -> {TRAIN}")
    print(f"Wrote {n_val} val   -> {VAL}")


if __name__ == "__main__":
    main()
