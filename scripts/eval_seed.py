"""Run the fine-tuned judge on the held-out HUMAN seed (never trained on) and
print predicted vs human scores -- the first real look at quality.

Usage: uv run python -m scripts.eval_seed
"""

from pathlib import Path

from judgekit import Sample

from src.data.schema import load_records
from src.judge.model import FineTunedJudge


def main() -> None:
    records = load_records(Path("eval/seed_human.jsonl"))
    judge = FineTunedJudge()
    print(f"{'id':10}{'faith pred/hum':>16}{'rel pred/hum':>16}")
    for r in records:
        j = judge.score(Sample(input=r.question, output=r.answer, context=r.context))
        pf, pr = j.scores["faithfulness"], j.scores["relevance"]
        print(f"{r.id:10}{f'{pf} / {r.faithfulness}':>16}{f'{pr} / {r.relevance}':>16}")


if __name__ == "__main__":
    main()
