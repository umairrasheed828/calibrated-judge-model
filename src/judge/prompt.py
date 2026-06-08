"""Turn the judging task into the exact chat messages the model trains on and is
prompted with at inference.

The rubric defined in task.py is encoded HERE into the system prompt, so the model
is trained to apply that exact definition. The same builders are reused at
inference/serving, so train and test prompts never drift.
"""

import json

from src.judge.task import AXES

_AXES_BLOCK = "\n".join(
    f"- {a.name} ({a.min_score}-{a.max_score}): {a.description}" for a in AXES
)

SYSTEM_PROMPT = (
    "You are a strict evaluator of research-QA answers. Score the ANSWER on each "
    "axis below as an integer, judging each axis independently.\n\n"
    f"{_AXES_BLOCK}\n\n"
    'Respond with ONLY a JSON object: {"faithfulness": <int>, "relevance": <int>}'
)

USER_TEMPLATE = (
    "QUESTION:\n{question}\n\n"
    "CONTEXT (the only allowed source of truth):\n{context}\n\n"
    "ANSWER:\n{answer}"
)


def build_messages(question: str, context: str, answer: str) -> list[dict[str, str]]:
    """The prompt the judge sees (system + user). Used at inference too."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": USER_TEMPLATE.format(
                question=question, context=context, answer=answer
            ),
        },
    ]


def target_completion(faithfulness: int, relevance: int) -> str:
    """The assistant text the judge is trained to produce."""
    return json.dumps({"faithfulness": faithfulness, "relevance": relevance})


def to_chat_example(
    question: str, context: str, answer: str, faithfulness: int, relevance: int
) -> dict[str, list[dict[str, str]]]:
    """Full training example: prompt messages + the target assistant JSON."""
    return {
        "messages": build_messages(question, context, answer)
        + [
            {
                "role": "assistant",
                "content": target_completion(faithfulness, relevance),
            }
        ]
    }
