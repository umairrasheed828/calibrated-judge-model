from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

load_dotenv(override=True)  # this project's .env is authoritative


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
    openai_api_key: str = (
        ""  # optional so the module imports cleanly in CI (no key there)
    )
    teacher_model: str = "gpt-4o-mini"
    mlflow_s3_bucket: str = ""  # reads MLFLOW_S3_BUCKET from .env
    mlflow_experiment: str = "calibrated-judge"
    hf_token: str = ""  # reads HF_TOKEN from .env


settings = Settings()
