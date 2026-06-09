---
base_model: Qwen/Qwen2.5-1.5B-Instruct
library_name: peft
license: apache-2.0
pipeline_tag: text-generation
tags:
- llm-as-judge
- evaluation
- calibration
- lora
- qlora
---

# Calibrated Research-QA Judge (Qwen2.5-1.5B + QLoRA)

A small, **calibrated LLM-as-judge** that scores research-QA answers on two axes —
**faithfulness** (is every claim supported by the context?) and **relevance** (does
it answer the question?) — each 1–5. A QLoRA adapter on `Qwen/Qwen2.5-1.5B-Instruct`,
distilled from a GPT-4o-mini teacher and evaluated with statistical rigor.

The point of this model is not raw quality — it's that its quality is **measured and
its confidence is calibrated**, with the evidence below.

## Intended use
Automated evaluation of grounded question-answering / RAG outputs, where a cheap,
local, private judge is preferable to a frontier API. Returns a JSON judgment:
`{"faithfulness": <1-5>, "relevance": <1-5>}`.

## How it was built
- **Base:** Qwen2.5-1.5B-Instruct, loaded in 4-bit (NF4).
- **Method:** QLoRA (LoRA r=16, α=32, dropout=0.05, all-linear targets), 3 epochs.
- **Data:** ~120 distilled examples with deliberate quality variation (faithful /
  unfaithful / off-topic), labelled by GPT-4o-mini; a 5-example human-labelled gold
  seed held out for testing.
- **Loss on completion only** — the model learns to *produce* the JSON judgment.

## Evaluation
Agreement (mean absolute error, 1–5 scale) and Cohen's κ:

| Comparison | Faithfulness MAE | Relevance MAE |
|---|---|---|
| Student vs **teacher** (val, n=24) | 0.46 | 0.21 |
| Student vs **human** (seed, n=5) | 1.20 (95% CI [0, 2.4]) | 0.20 |
| GPT-4o-mini vs **human** (seed, n=5) | 0.60 (95% CI [0, 1.4]) | 0.20 |

- **Relevance: matches GPT-4o-mini** (MAE 0.20, κ 0.58 for both).
- **Faithfulness:** point estimate trails GPT-4o-mini, but with n=5 the bootstrap
  CIs overlap entirely — the difference is **not statistically resolvable** on this
  set. The model learned the teacher well (val MAE 0.46); the open question is
  human-grounded faithfulness accuracy.

### Calibration (faithfulness confidence)
Confidence `P(faithful) = P(score ≥ 4)` read from the score-token logits, vs the
binary outcome (label ≥ 4):

| | ECE | Brier | Temperature |
|---|---|---|---|
| Raw | 0.142 | 0.088 | — |
| After temperature scaling | **0.070** | — | T = 1.25 |

Temperature scaling roughly **halves** ECE. Apply **T = 1.25** to the confidence at
inference for calibrated probabilities.

## Limitations
- **Lenient on faithfulness vs humans** on the (small) human set — inherited from a
  lenient teacher. Treat its faithfulness scores as an upper bound.
- Human evaluation set is tiny (n=5); numbers are indicative, not settled.
- Training data is synthetic (teacher-labelled); domain is AI/ML research QA.

## Usage
```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct", device_map="auto")
model = PeftModel.from_pretrained(base, "umairrasheed828/calibrated-research-qa-judge")
tok = AutoTokenizer.from_pretrained("umairrasheed828/calibrated-research-qa-judge")
# Prompt with the system rubric + QUESTION/CONTEXT/ANSWER; model returns JSON scores.
```

## License
Apache-2.0 (inherits from the Qwen2.5 base).