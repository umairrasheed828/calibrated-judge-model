"""Load the fine-tuned adapter and expose it as a judgekit-compatible Judge.

The LoRA adapter sits on the frozen 4-bit base. We wrap inference (build prompt ->
generate -> parse JSON) behind judgekit's Judge protocol -- an object with
.score(sample) -> Judgment -- so the same library that benchmarks it can call it,
and the serving API reuses this class verbatim.
"""

import json
import re

import torch
from judgekit import Judgment, Sample
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from src.judge.prompt import build_messages
from typing import Any, cast

BASE_MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
DEFAULT_ADAPTER = "models/judge-adapter"


def _parse_scores(text: str) -> dict[str, int]:
    """Pull {faithfulness, relevance} out of the model's JSON; fall back to digits."""
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end != -1:
        try:
            data = json.loads(text[start : end + 1])
            return {
                "faithfulness": int(data["faithfulness"]),
                "relevance": int(data["relevance"]),
            }
        except (ValueError, KeyError, TypeError):
            pass
    nums = [int(n) for n in re.findall(r"[1-5]", text)]
    return {
        "faithfulness": nums[0] if nums else 1,
        "relevance": nums[1] if len(nums) > 1 else 1,
    }


class FineTunedJudge:
    """judgekit-compatible judge backed by the fine-tuned adapter."""

    def __init__(self, adapter_dir: str = DEFAULT_ADAPTER) -> None:
        quant = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        base = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_ID, quantization_config=quant, dtype="auto", device_map="auto"
        )
        self.model = PeftModel.from_pretrained(base, adapter_dir)
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(adapter_dir)

    def score(self, sample: Sample) -> Judgment:
        messages = build_messages(sample.input, sample.context or "", sample.output)
        inputs = cast(
            Any,
            self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=True,
            ),
        ).to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=20, do_sample=False)
        text = cast(
            str,
            self.tokenizer.decode(
                out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
            ),
        )
        return Judgment(scores=_parse_scores(text), rationale=text.strip())

    def _digit_distribution(
        self, messages: list[dict[str, str]], prefix: str
    ) -> list[float]:
        """P(score=1..5) from the model's logits at the digit right after `prefix`."""
        prompt = cast(
            str,
            self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            ),
        )
        ids = cast(Any, self.tokenizer(prompt + prefix, return_tensors="pt")).to(
            self.model.device
        )
        with torch.no_grad():
            logits = cast(Any, self.model(**ids)).logits[0, -1]
        digit_ids = [
            self.tokenizer.encode(str(d), add_special_tokens=False)[0]
            for d in range(1, 6)
        ]
        probs = cast(Any, torch.softmax(logits[digit_ids], dim=-1))
        return [float(p) for p in probs]

    def faithfulness_confidence(self, sample: Sample) -> float:
        """P(faithful) = P(faithfulness score >= 4), from the digit's logits."""
        messages = build_messages(sample.input, sample.context or "", sample.output)
        dist = self._digit_distribution(messages, '{"faithfulness": ')
        return dist[3] + dist[4]
