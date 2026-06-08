from pathlib import Path

import pytest

from src.data.schema import JudgeRecord, load_records, save_records


def _valid() -> dict:
    return {
        "id": "ex1",
        "question": "What is RAG?",
        "context": "RAG retrieves documents and grounds generation on them.",
        "answer": "RAG retrieves documents and grounds the answer in them.",
        "faithfulness": 5,
        "relevance": 5,
        "label_source": "human",
    }


def test_valid_record_builds() -> None:
    rec = JudgeRecord(**_valid())
    assert rec.faithfulness == 5
    assert rec.label_source == "human"


def test_out_of_range_score_rejected() -> None:
    with pytest.raises(ValueError):
        JudgeRecord(**(_valid() | {"faithfulness": 7}))


def test_unknown_label_source_rejected() -> None:
    with pytest.raises(ValueError):
        JudgeRecord(**(_valid() | {"label_source": "intern"}))


def test_round_trip(tmp_path: Path) -> None:
    recs = [JudgeRecord(**_valid())]
    path = tmp_path / "x.jsonl"
    save_records(recs, path)
    assert load_records(path) == recs
