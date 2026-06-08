"""MLflow wiring: run metadata stays local (./mlruns), artifacts go to S3.

The split is deliberate -- params/metrics are tiny and fine on disk, while the
artifacts (adapter, plots, model card) need a durable home. The S3 artifact
location is fixed once, when the experiment is created.
"""

import mlflow

from src.config import settings


def artifact_uri() -> str:
    return f"s3://{settings.mlflow_s3_bucket}/mlflow"


def setup_mlflow() -> str:
    """Create-or-activate the experiment (artifacts -> S3); return its name."""
    name = settings.mlflow_experiment
    if mlflow.get_experiment_by_name(name) is None:
        mlflow.create_experiment(name, artifact_location=artifact_uri())
    mlflow.set_experiment(name)
    return name
