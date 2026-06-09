"""Calibrate the judge's FAITHFULNESS confidence (the hard axis) and plot it.

The judge emits a discrete 1-5 score, but the score TOKEN carries a probability:
P(faithful) = P(score >= 4), read from the model's logits at the faithfulness
digit. We compare that to the binary outcome (label >= 4) on val, measure
miscalibration (ECE, Brier), fix it (temperature scaling), and plot reliability
before/after. Logged to MLflow (metrics + plot -> S3).

Usage: uv run python -m scripts.calibrate
"""

from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
from judgekit import Sample, apply_temperature, calibration_report, reliability_bins

from src.data.schema import load_records
from src.judge.model import FineTunedJudge
from src.tracking import setup_mlflow

N_BINS = 5  # small set -> few bins


def main() -> None:
    val = load_records(Path("datasets/val.jsonl"))
    judge = FineTunedJudge()

    confidences: list[float] = []
    outcomes: list[int] = []
    for r in val:
        c = judge.faithfulness_confidence(
            Sample(input=r.question, output=r.answer, context=r.context)
        )
        confidences.append(c)
        outcomes.append(1 if r.faithfulness >= 4 else 0)

    print(
        f"mean confidence={sum(confidences) / len(confidences):.3f} "
        f"actual faithful rate={sum(outcomes) / len(outcomes):.3f}"
    )
    report = calibration_report(confidences, outcomes, n_bins=N_BINS)
    print(" ", report.summary())

    before = reliability_bins(confidences, outcomes, N_BINS)
    after = reliability_bins(
        apply_temperature(confidences, report.temperature), outcomes, N_BINS
    )

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="perfect")
    for bins, lbl in ((before, "raw"), (after, f"T={report.temperature:.2f}")):
        xs = [b.avg_confidence for b in bins if b.count]
        ys = [b.accuracy for b in bins if b.count]
        ax.plot(xs, ys, "o-", label=lbl)
    ax.set_xlabel("mean confidence")
    ax.set_ylabel("accuracy")
    ax.set_title("Faithfulness reliability (before/after temperature)")
    ax.legend()
    plot_path = Path("reliability.png")
    fig.savefig(plot_path, dpi=120, bbox_inches="tight")

    setup_mlflow()
    with mlflow.start_run(run_name="calibration-faithfulness"):
        mlflow.log_metric("ece_before", report.ece)
        mlflow.log_metric("ece_after", report.ece_after)
        mlflow.log_metric("brier", report.brier)
        mlflow.log_metric("temperature", report.temperature)
        mlflow.log_artifact(str(plot_path))
    plot_path.unlink()
    print(
        f"\nECE {report.ece:.3f} -> {report.ece_after:.3f} (T={report.temperature:.2f})"
    )
    print("Logged calibration + reliability plot to MLflow (S3).")


if __name__ == "__main__":
    main()
