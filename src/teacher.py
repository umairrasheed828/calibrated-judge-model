"""The teacher: GPT-4o-mini as a plain-text completion callable.

Exposed as `complete(prompt) -> str` -- the exact shape judgekit's LLMJudge
expects -- so the same teacher labels the training set now AND serves as the
benchmark baseline later, through one code path. OpenAI is imported lazily so
this module imports without a key (CI stays green).
"""

from collections.abc import Callable

from src.config import settings


def get_completer(temperature: float = 0.0) -> Callable[[str], str]:
    from openai import OpenAI  # lazy: keeps the module importable with no key

    client = OpenAI(api_key=settings.openai_api_key)

    def complete(prompt: str) -> str:
        resp = client.chat.completions.create(
            model=settings.teacher_model,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.choices[0].message.content or ""

    return complete
