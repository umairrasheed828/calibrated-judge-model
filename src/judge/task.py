"""The judging task: what this model scores, and exactly what 'good' means.

Reuses the faithfulness/relevance axes calibrated in P1/P2/judgekit, so the
fine-tuned judge speaks the same vocabulary as the library that evaluates it.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class JudgeAxis:
    name: str
    description: str
    min_score: int = 1
    max_score: int = 5


FAITHFULNESS = JudgeAxis(
    name="faithfulness",
    description=(
        "Is EVERY claim in the answer supported by the context? "
        "5: every claim is grounded in the context. "
        "3: mostly grounded, one minor unsupported detail. "
        "1: contains a clearly fabricated or contradicted claim. "
        "Judge ONLY grounding here — a claim can be true in the world but "
        "still unfaithful if the context does not support it."
    ),
)

RELEVANCE = JudgeAxis(
    name="relevance",
    description=(
        "Does the answer address the question that was asked? "
        "5: directly and fully answers it. "
        "3: partially answers it, or answers a narrower version. "
        "1: answers a different question or is off-topic. "
        "Judge ONLY topical fit here — incorrectness is a FAITHFULNESS "
        "problem, never a relevance one."
    ),
)

AXES: list[JudgeAxis] = [FAITHFULNESS, RELEVANCE]
