"""Smoke test: prove MLflow logs run metadata locally and an artifact to S3.

Usage: uv run python -m scripts.check_mlflow
"""

from pathlib import Path

import boto3
import mlflow

from src.tracking import setup_mlflow


def main() -> None:
    setup_mlflow()
    with mlflow.start_run(run_name="s3-smoke-test") as run:
        mlflow.log_param("hello", "world")
        mlflow.log_metric("dummy_metric", 0.42)
        note = Path("smoke.txt")
        note.write_text("artifact upload works", encoding="utf-8")
        mlflow.log_artifact(str(note))
        note.unlink()
        artifact_root = mlflow.get_artifact_uri()
    print("run_id:", run.info.run_id)
    print("artifact_uri:", artifact_root)

    assert artifact_root.startswith("s3://"), "artifacts are NOT going to S3"
    bucket, _, prefix = artifact_root.removeprefix("s3://").partition("/")
    resp = boto3.client("s3").list_objects_v2(
        Bucket=bucket, Prefix=f"{prefix}/smoke.txt"
    )
    found = any(o["Key"].endswith("smoke.txt") for o in resp.get("Contents", []))
    print("artifact present in S3:", found)


if __name__ == "__main__":
    main()
