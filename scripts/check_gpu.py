"""Smoke test: confirm the GPU stack works and the 4-bit base model fits.

Loads Qwen2.5-1.5B-Instruct in 4-bit (NF4) and runs one tiny generation. If
torch.cuda.is_available() is False, bitsandbytes will fall back to CPU -- that
means torch is the CPU build, so re-check the pytorch index.

Usage: uv run python -m scripts.check_gpu
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"


def main() -> None:
    print(f"torch {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"device: {torch.cuda.get_device_name(0)}")

    quant = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, quantization_config=quant, dtype="auto", device_map="auto"
    )
    print(
        f"loaded {MODEL_ID} in 4-bit -- footprint ~{model.get_memory_footprint() / 1e9:.2f} GB"
    )

    messages = [{"role": "user", "content": "Reply with exactly: gpu ok"}]
    inputs = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
    ).to(model.device)
    out = model.generate(**inputs, max_new_tokens=10)
    new_tokens = out[0][inputs["input_ids"].shape[1] :]
    print("generation:", tokenizer.decode(new_tokens, skip_special_tokens=True))


if __name__ == "__main__":
    main()
