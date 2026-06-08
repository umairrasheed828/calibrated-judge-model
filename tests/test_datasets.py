from pathlib import Path

from src.data.schema import load_records

TRAIN = Path("datasets/train.jsonl")
VAL = Path("datasets/val.jsonl")


def test_splits_exist_and_are_distilled() -> None:
    for path in (TRAIN, VAL):
        assert path.exists(), "run: uv run python -m scripts.label_pool"
        records = load_records(path)
        assert records
        assert all(r.label_source == "gpt-4o-mini" for r in records)
