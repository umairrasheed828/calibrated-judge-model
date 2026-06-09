"""Benchmark the fine-tuned judge vs the GPT-4o-mini teacher, through judgekit.

Per-axis agreement (MAE + 95% bootstrap CI, Cohen's kappa) for:
  - student vs TEACHER on val   (did the small model learn the teacher?)
  - student vs HUMAN on the seed (the honest gold test)
  - GPT-4o-mini vs HUMAN on seed (the frontier baseline)
plus latency and cost. Logged to MLflow (metrics + comparison table -> S3).

Usage: uv run python -m scripts.benchmark
"""

import json
import time
from pathlib import Path

import mlflow
from judgekit import Axis, LLMJudge, Sample, bootstrap_ci, cohens_kappa

from src.data.schema import load_records
from src.judge.model import FineTunedJudge
from src.judge.task import AXES
from src.teacher import get_completer
from src.tracking import setup_mlflow

AXIS_NAMES = [a.name for a in AXES]


def score_all(judge, records):
    """Run a judge over records; return per-axis predictions and mean latency."""
    preds = {a: [] for a in AXIS_NAMES}
    times = []
    for r in records:
        t0 = time.perf_counter()
        j = judge.score(Sample(input=r.question, output=r.answer, context=r.context))
        times.append(time.perf_counter() - t0)
        for a in AXIS_NAMES:
            preds[a].append(j.scores[a])
    return preds, sum(times) / len(times)


def labels(records):
    return {a: [getattr(r, a) for r in records] for a in AXIS_NAMES}


def agreement(pred, gold):
    """Per-axis MAE (+95% CI) and Cohen's kappa."""
    out = {}
    for a in AXIS_NAMES:
        errs = [abs(p - g) for p, g in zip(pred[a], gold[a])]
        mae, lo, hi = bootstrap_ci(errs)
        out[a] = {
            "mae": round(mae, 3),
            "mae_ci95": [round(lo, 3), round(hi, 3)],
            "kappa": round(cohens_kappa(pred[a], gold[a]), 3),
        }
    return out


def main() -> None:
    val = load_records(Path("datasets/val.jsonl"))
    seed = load_records(Path("eval/seed_human.jsonl"))

    student = FineTunedJudge()
    teacher = LLMJudge(
        [Axis(a.name, a.description, a.min_score, a.max_score) for a in AXES],
        get_completer(),
    )

    s_val_pred, _ = score_all(student, val)
    s_seed_pred, s_lat = score_all(student, seed)
    t_seed_pred, t_lat = score_all(teacher, seed)

    results = {
        "student_vs_teacher_val": agreement(s_val_pred, labels(val)),
        "student_vs_human_seed": agreement(s_seed_pred, labels(seed)),
        "gpt4o_mini_vs_human_seed": agreement(t_seed_pred, labels(seed)),
        "latency_s_per_call": {
            "student_local": round(s_lat, 3),
            "gpt4o_mini_api": round(t_lat, 3),
        },
        "cost": {
            "student": "free (local, ~$0 marginal after training)",
            "gpt4o_mini": "per-token API cost",
        },
    }
    print(json.dumps(results, indent=2))

    setup_mlflow()
    with mlflow.start_run(run_name="benchmark-vs-gpt4o-mini"):
        for tag, block in (
            ("student_human", results["student_vs_human_seed"]),
            ("gpt4o_human", results["gpt4o_mini_vs_human_seed"]),
            ("student_teacher", results["student_vs_teacher_val"]),
        ):
            for a in AXIS_NAMES:
                mlflow.log_metric(f"{tag}_{a}_mae", block[a]["mae"])
                mlflow.log_metric(f"{tag}_{a}_kappa", block[a]["kappa"])
        mlflow.log_metric("latency_student_s", s_lat)
        mlflow.log_metric("latency_gpt4o_s", t_lat)
        out = Path("benchmark.json")
        out.write_text(json.dumps(results, indent=2), encoding="utf-8")
        mlflow.log_artifact(str(out))
        out.unlink()
    print("\nLogged benchmark to MLflow (comparison table -> S3).")


if __name__ == "__main__":
    main()
