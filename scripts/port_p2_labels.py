"""Port P2 (multi-agent-analyst) human calibration labels into P4's schema.

P2's calibration_set.jsonl already holds everything we need: a question, the
fact-checked notes (our `context`), the brief being judged (our `answer`), and
human faithfulness/relevance labels. We carry those over verbatim as the gold,
human-labeled seed -- the immovable ground truth the whole project anchors on.

Usage: uv run python -m scripts.port_p2_labels
"""

import json
from pathlib import Path

from src.data.schema import JudgeRecord, save_records

RAW = Path("data/raw/p2_calibration.jsonl")
OUT = Path("eval/seed_human.jsonl")


def port() -> list[JudgeRecord]:
    if not RAW.exists():
        raise FileNotFoundError(
            f"{RAW} not found. Copy P2's eval/calibration_set.jsonl there first."
        )
    records: list[JudgeRecord] = []
    for line in RAW.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        records.append(
            JudgeRecord(
                id=f"p2_{row['id']}",  # provenance: where this gold label came from
                question=row["question"],
                context=row["notes"],
                answer=row["brief"],
                faithfulness=int(row["human_faithfulness"]),
                relevance=int(row["human_relevance"]),
                label_source="human",
            )
        )
    return records


def main() -> None:
    records = port()
    save_records(records, OUT)
    print(f"Ported {len(records)} human-labeled records -> {OUT}")
    for axis in ("faithfulness", "relevance"):
        vals = [getattr(r, axis) for r in records]
        dist = {s: vals.count(s) for s in sorted(set(vals))}
        print(f"  {axis}: mean={sum(vals) / len(vals):.2f} dist={dist}")


if __name__ == "__main__":
    main()
