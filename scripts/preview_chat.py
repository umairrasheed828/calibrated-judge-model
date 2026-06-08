"""Preview one formatted training example -- exactly what the model trains on.

Usage: uv run python -m scripts.preview_chat
"""

from pathlib import Path

from src.data.schema import load_records
from src.judge.prompt import to_chat_example


def main() -> None:
    rec = load_records(Path("datasets/train.jsonl"))[0]
    example = to_chat_example(
        rec.question, rec.context, rec.answer, rec.faithfulness, rec.relevance
    )
    for msg in example["messages"]:
        print(f"\n=== {msg['role'].upper()} ===")
        print(msg["content"])


if __name__ == "__main__":
    main()
