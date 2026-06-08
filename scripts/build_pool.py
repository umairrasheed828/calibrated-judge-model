"""Build an UNLABELED pool of (question, context, answer) triples with deliberate
quality variation, so the judge later sees the FULL score range -- not just 5/5.

Quality is varied BY CONSTRUCTION across three intents (good / unfaithful /
off_topic). We only generate items here; the teacher LABELS them next (that is the
distillation). The `intended` tag is an audit signal only -- never a training label.

Usage: uv run python -m scripts.build_pool
"""

import json
from pathlib import Path

from openai import OpenAI
from pydantic import BaseModel

from src.config import settings

OUT = Path("data/pool.jsonl")
REPEATS = 2  # bump this to grow the training set before a serious run
TEMPERATURE = 0.7  # variety, not determinism -- this is generation, not judging

TOPICS = [
    "retrieval-augmented generation",
    "the transformer architecture",
    "self-attention",
    "word and sentence embeddings",
    "vector databases",
    "BM25 keyword retrieval",
    "reranking with cross-encoders",
    "reinforcement learning from human feedback",
    "model quantization",
    "LoRA and parameter-efficient fine-tuning",
    "fine-tuning versus prompting",
    "context windows in large language models",
    "tokenization",
    "LLM-as-a-judge evaluation",
    "calibration and expected calibration error",
    "hallucination in language models",
    "AI agents and tool use",
    "chain-of-thought prompting",
    "knowledge distillation",
    "mixture-of-experts models",
]

INTENTS = {
    "good": (
        "a concise answer that is FULLY supported by the context and directly "
        "answers the question."
    ),
    "unfaithful": (
        "a concise, confident answer that addresses the question but includes at "
        "least one specific claim that is NOT supported by the context -- a "
        "plausible-sounding fabrication."
    ),
    "off_topic": (
        "a fluent, concise answer that does NOT address the question -- it talks "
        "about something else related to the topic instead."
    ),
}

GEN_PROMPT = """You are creating evaluation data for a research-QA judge.

Topic: {topic}

Produce:
- question: a specific question about the topic.
- context: 2-4 sentences of factual background a good answer should rely on.
- answer: {intent}
"""


class GeneratedItem(BaseModel):
    question: str
    context: str
    answer: str


def main() -> None:
    client = OpenAI(api_key=settings.openai_api_key)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    counts: dict[str, int] = {k: 0 for k in INTENTS}
    with OUT.open("w", encoding="utf-8") as f:
        for _ in range(REPEATS):
            for topic in TOPICS:
                for intent, instruction in INTENTS.items():
                    completion = client.chat.completions.parse(
                        model=settings.teacher_model,
                        temperature=TEMPERATURE,
                        messages=[
                            {
                                "role": "user",
                                "content": GEN_PROMPT.format(
                                    topic=topic, intent=instruction
                                ),
                            }
                        ],
                        response_format=GeneratedItem,
                    )
                    item = completion.choices[0].message.parsed
                    if item is None:
                        continue
                    f.write(
                        json.dumps(
                            {
                                "question": item.question,
                                "context": item.context,
                                "answer": item.answer,
                                "topic": topic,
                                "intended": intent,
                            }
                        )
                        + "\n"
                    )
                    n += 1
                    counts[intent] += 1
                    if n % 20 == 0:
                        print(f"  ...{n} items")
    print(f"Generated {n} items -> {OUT}")
    print(f"  by intent: {counts}")


if __name__ == "__main__":
    main()
