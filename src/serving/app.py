from __future__ import annotations
import logging, time
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from src.security.guardrails import InputGuardrail, OutputGuardrail

logger = logging.getLogger(__name__)
input_guardrail = InputGuardrail()
output_guardrail = OutputGuardrail()
START_TIME = time.time()


class PredictRequest(BaseModel):
    transaction_id: str
    features: dict[str, float]

class PredictResponse(BaseModel):
    transaction_id: str
    fraud_probability: float
    decision: str
    model_version: str

class AskRequest(BaseModel):
    query: str
    session_id: str | None = None

class AskResponse(BaseModel):
    answer: str
    sources: list[str]
    iterations: int
    had_pii: bool

class HealthResponse(BaseModel):
    status: str
    model_version: str
    uptime_s: float


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    logger.info("API iniciada")
    yield
    logger.info("API encerrada")


app = FastAPI(title="Fraud Detection API", version="1.0.0", lifespan=lifespan)


@app.middleware("http")
async def logging_middleware(request: Request, call_next: Any) -> Any:
    start = time.time()
    response = await call_next(request)
    duration = time.time() - start
    logger.info("Request", extra={"path": request.url.path, "status": response.status_code, "ms": round(duration * 1000, 2)})
    return response


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(status="healthy", model_version="1.0.0", uptime_s=round(time.time() - START_TIME, 2))


@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest) -> PredictResponse:
    v14 = request.features.get("V14", 0.0)
    amount = request.features.get("Amount", 0.0)
    fraud_prob = max(0.0, min(1.0, 0.5 - v14 * 0.05 + amount / 100000))
    decision = "BLOCKED" if fraud_prob >= 0.35 else "APPROVED"
    logger.info("Predicao", extra={"tx": request.transaction_id, "prob": fraud_prob})
    return PredictResponse(
        transaction_id=request.transaction_id,
        fraud_probability=fraud_prob,
        decision=decision,
        model_version="1.0.0",
    )


@app.post("/ask", response_model=AskResponse)
async def ask(request: AskRequest) -> AskResponse:
    gr = input_guardrail.check(request.query)
    if not gr["allowed"]:
        raise HTTPException(status_code=400, detail="Requisicao bloqueada: " + gr["reason"])
    mock_answer = "Metricas: AUC-ROC 0.9743, F1 0.8821. Sem drift detectado nas ultimas 24h."
    sanitized, had_pii = output_guardrail.apply(mock_answer)
    return AskResponse(answer=sanitized, sources=["fraud_metrics", "drift_status"], iterations=2, had_pii=had_pii)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
