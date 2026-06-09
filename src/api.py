"""Minimal serving API for the calibrated research-QA judge.

POST /judge -> {faithfulness, relevance, faithfulness_confidence}, where the
confidence is temperature-CALIBRATED (T from Step 14) -- the differentiator in
the product itself. The model loads once at startup.
"""

from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, HTTPException
from judgekit import Sample, apply_temperature
from pydantic import BaseModel

from src.judge.model import FineTunedJudge

FAITHFULNESS_TEMPERATURE = 1.25  # fitted in Step 14; re-fit if you retrain

_judge: FineTunedJudge | None = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    global _judge
    _judge = FineTunedJudge()
    yield


app = FastAPI(title="Calibrated Research-QA Judge", lifespan=lifespan)


class JudgeRequest(BaseModel):
    question: str
    context: str
    answer: str


class JudgeResponse(BaseModel):
    faithfulness: int
    relevance: int
    faithfulness_confidence: float  # calibrated P(faithful)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_loaded": _judge is not None}


@app.post("/judge", response_model=JudgeResponse)
def judge(req: JudgeRequest) -> JudgeResponse:
    if _judge is None:
        raise HTTPException(status_code=503, detail="model not loaded")
    if not req.question.strip() or not req.answer.strip():
        raise HTTPException(status_code=400, detail="question and answer are required")
    sample = Sample(input=req.question, output=req.answer, context=req.context)
    judgment = _judge.score(sample)
    raw = _judge.faithfulness_confidence(sample)
    calibrated = apply_temperature([raw], FAITHFULNESS_TEMPERATURE)[0]
    return JudgeResponse(
        faithfulness=judgment.scores["faithfulness"],
        relevance=judgment.scores["relevance"],
        faithfulness_confidence=round(calibrated, 3),
    )
