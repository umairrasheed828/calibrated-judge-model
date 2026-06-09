"""QLoRA fine-tune Qwen2.5-1.5B-Instruct into a research-QA judge.

Loads the base in 4-bit (NF4), attaches LoRA adapters, and trains on the distilled
prompt->completion examples with loss on the COMPLETION ONLY (the model learns to
PRODUCE the JSON judgment, not echo the rubric). Tracked in MLflow: params + loss
curve locally, the adapter as an artifact in S3.

Usage: uv run python -m scripts.train
"""

from pathlib import Path

import mlflow
import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

from src.data.schema import JudgeRecord, load_records
from src.judge.prompt import build_messages, target_completion
from src.tracking import setup_mlflow

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
ADAPTER_DIR = Path("models/judge-adapter")
OUTPUT_DIR = "models/qlora-run"

EPOCHS = 3
BATCH_SIZE = 2
GRAD_ACCUM = 4
LR = 2e-4
MAX_LENGTH = 1024
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05


def to_example(rec: JudgeRecord) -> dict:
    """Conversational prompt-completion: loss falls on the completion only."""
    return {
        "prompt": build_messages(rec.question, rec.context, rec.answer),
        "completion": [
            {
                "role": "assistant",
                "content": target_completion(rec.faithfulness, rec.relevance),
            }
        ],
    }


def load_split(path: str) -> Dataset:
    return Dataset.from_list([to_example(r) for r in load_records(Path(path))])


def main() -> None:
    train_ds = load_split("datasets/train.jsonl")
    val_ds = load_split("datasets/val.jsonl")

    quant = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, quantization_config=quant, dtype="auto", device_map="auto"
    )
    model.config.use_cache = False

    lora = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules="all-linear",
    )

    args = SFTConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LR,
        max_length=MAX_LENGTH,
        completion_only_loss=True,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="adamw_torch",
        logging_steps=2,
        eval_strategy="epoch",
        save_strategy="no",
        report_to="none",  # we log to MLflow manually, into OUR S3-backed run
    )

    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        peft_config=lora,
        processing_class=tokenizer,
    )

    setup_mlflow()
    with mlflow.start_run(run_name="qlora-qwen2.5-1.5b"):
        mlflow.log_params(
            {
                "base_model": MODEL_ID,
                "lora_r": LORA_R,
                "lora_alpha": LORA_ALPHA,
                "lora_dropout": LORA_DROPOUT,
                "epochs": EPOCHS,
                "batch_size": BATCH_SIZE,
                "grad_accum": GRAD_ACCUM,
                "learning_rate": LR,
                "max_length": MAX_LENGTH,
                "train_size": len(train_ds),
                "val_size": len(val_ds),
            }
        )
        trainer.train()

        for entry in trainer.state.log_history:
            step = int(entry.get("step", 0))
            if "loss" in entry:
                mlflow.log_metric("train_loss", entry["loss"], step=step)
            if "eval_loss" in entry:
                mlflow.log_metric("eval_loss", entry["eval_loss"], step=step)

        ADAPTER_DIR.mkdir(parents=True, exist_ok=True)
        trainer.save_model(str(ADAPTER_DIR))
        tokenizer.save_pretrained(str(ADAPTER_DIR))
        mlflow.log_artifacts(str(ADAPTER_DIR), artifact_path="judge-adapter")
        print(f"\nAdapter saved to {ADAPTER_DIR} and logged to MLflow (S3).")


if __name__ == "__main__":
    main()
