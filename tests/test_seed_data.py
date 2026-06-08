from pathlib import Path

from src.data.schema import load_records

SEED = Path("eval/seed_human.jsonl")


def test_seed_exists_and_is_all_human() -> None:
    assert SEED.exists(), "run: uv run python -m scripts.port_p2_labels"
    records = load_records(SEED)
    assert records, "seed must not be empty"
    assert all(r.label_source == "human" for r in records)
