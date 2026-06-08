"""Verify AWS credentials and bucket access before wiring MLflow to S3.

Usage: uv run python -m scripts.check_s3
"""

import os

import boto3

from src.config import settings  # importing this loads .env (AWS_* into os.environ)

assert settings  # keep the import (it triggers load_dotenv)


def main() -> None:
    bucket = os.environ["MLFLOW_S3_BUCKET"]
    s3 = boto3.client("s3")
    names = [b["Name"] for b in s3.list_buckets()["Buckets"]]
    print("Reachable buckets:", names)
    print(f"Target bucket {bucket!r} found:", bucket in names)


if __name__ == "__main__":
    main()
