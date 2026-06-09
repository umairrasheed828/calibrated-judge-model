"""Publish the LoRA adapter + model card to the HuggingFace Hub (public proof).

Usage: uv run python -m scripts.publish
"""

import shutil
from pathlib import Path

from huggingface_hub import HfApi

from src.config import settings

REPO_ID = "umairrasheed828/calibrated-research-qa-judge"
ADAPTER_DIR = Path("models/judge-adapter")
CARD = Path("MODEL_CARD.md")


def main() -> None:
    if not settings.hf_token:
        raise SystemExit(
            "Set HF_TOKEN in .env (HF Settings -> Access Tokens -> Write)."
        )
    if not ADAPTER_DIR.exists():
        raise SystemExit(f"{ADAPTER_DIR} not found -- run training first.")

    shutil.copy(CARD, ADAPTER_DIR / "README.md")  # the card IS the HF repo README
    api = HfApi(token=settings.hf_token)
    api.create_repo(REPO_ID, repo_type="model", private=False, exist_ok=True)
    api.upload_folder(folder_path=str(ADAPTER_DIR), repo_id=REPO_ID, repo_type="model")
    print(f"Published -> https://huggingface.co/{REPO_ID}")


if __name__ == "__main__":
    main()
