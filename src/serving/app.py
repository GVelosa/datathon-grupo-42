"""FastAPI serving layer: endpoints /predict, /ask e /health."""

from __future__ import annotations

import logging
import os
import time
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request, Response
from pydantic import BaseModel, Field

from src.security.guardrails import InputGuardrail, OutputGuardrail

# Carrega .env da raiz do projeto independente do diretório de execução
load_dotenv(Path(__file__).parent.parent.parent / ".env")

logger = logging.getLogger(__name__)

START_TIME: float = time.time()
_agent: Any = None
_input_guardrail: InputGuardrail = InputGuardrail()
_output_guardrail: OutputGuardrail = OutputGuardrail()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Gerencia o ciclo de vida da API: inicializa o agente na subida.

    Yields:
        Controle para o FastAPI durante o tempo de vida da aplicação.
    """
    global _agent
    try:
        from src.agent.react_agent import BankHealthAgent

        _agent = BankHealthAgent(
            api_key=os.getenv("ANTHROPIC_API_KEY"),
            model=os.getenv("AGENT_MODEL", "claude-haiku-4-5-20251001"),
        )
        logger.info("BankHealthAgent inicializado com sucesso")
    except Exception as exc:
        logger.warning(
            "BankHealthAgent não inicializado (ANTHROPIC_API_KEY ausente?)",
            extra={"error": str(exc)},
        )
        _agent = None

    logger.info("API iniciada", extra={"start_time": START_TIME})
    yield
    logger.info("API encerrada")


app = FastAPI(
    title="Fraud Detection API",
    version="1.0.0",
    description="API de detecção de fraude com agente ReAct de saúde bancária.",
    lifespan=lifespan,
)

from prometheus_client import make_asgi_app as _make_asgi_app  # noqa: E402

app.mount("/metrics", _make_asgi_app())


@app.middleware("http")
async def logging_middleware(request: Request, call_next: Any) -> Response:
    """Middleware de logging estruturado para todas as requisições.

    Args:
        request: Requisição HTTP recebida.
        call_next: Próximo handler na cadeia.

    Returns:
        Response com headers e logging de latência.
    """
    start = time.time()
    response: Response = await call_next(request)
    duration_ms = round((time.time() - start) * 1000, 2)
    logger.info(
        "HTTP request",
        extra={
            "method": request.method,
            "path": request.url.path,
            "status": response.status_code,
            "duration_ms": duration_ms,
        },
    )
    return response


# ---------------------------------------------------------------------------
# Schemas Pydantic
# ---------------------------------------------------------------------------


class PredictRequest(BaseModel):
    """Schema de entrada para predição de fraude clássica.

    Attributes:
        transaction_id: Identificador único da transação.
        features: Dicionário de features numéricas (V1-V28, Amount, etc.).
    """

    transaction_id: str = Field(..., description="ID único da transação.")
    features: dict[str, float] = Field(..., description="Features numéricas da transação.")


class PredictResponse(BaseModel):
    """Schema de saída da predição de fraude.

    Attributes:
        transaction_id: ID da transação avaliada.
        fraud_probability: Probabilidade de fraude em [0, 1].
        decision: Decisão binária: APPROVED ou BLOCKED.
        model_version: Versão do modelo em produção.
    """

    transaction_id: str
    fraud_probability: float
    decision: str
    model_version: str


class AskRequest(BaseModel):
    """Schema de entrada para o agente ReAct de saúde bancária.

    Attributes:
        query: Pergunta em linguagem natural para o agente.
        session_id: ID de sessão opcional para rastreamento.
    """

    query: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="Pergunta em linguagem natural.",
    )
    session_id: str | None = Field(default=None, description="ID de sessão para rastreamento.")


class AskResponse(BaseModel):
    """Schema de saída do agente ReAct.

    Attributes:
        answer: Resposta final sanitizada (sem PII).
        tools_used: Ferramentas invocadas durante o raciocínio.
        iterations: Número de ciclos Thought/Action/Observation.
        had_pii: Se PII foi detectado e removido do output.
    """

    answer: str
    tools_used: list[str]
    iterations: int
    had_pii: bool


class HealthResponse(BaseModel):
    """Schema de saída do health check.

    Attributes:
        status: Estado da API (healthy/degraded).
        model_version: Versão do modelo em produção.
        agent_ready: Se o agente ReAct está inicializado.
        uptime_s: Tempo de operação em segundos desde o start.
    """

    status: str
    model_version: str
    agent_ready: bool
    uptime_s: float


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/health", response_model=HealthResponse, tags=["infra"])
async def health() -> HealthResponse:
    """Verifica o estado da API e disponibilidade do agente.

    Returns:
        HealthResponse com status, versão do modelo e uptime.
    """
    return HealthResponse(
        status="healthy",
        model_version="1.0.0",
        agent_ready=_agent is not None,
        uptime_s=round(time.time() - START_TIME, 2),
    )


@app.post("/predict", response_model=PredictResponse, tags=["ml"])
async def predict(request: PredictRequest) -> PredictResponse:
    """Prediz a probabilidade de fraude de uma transação usando o modelo clássico.

    Aplica heurística baseada em V14 e Amount como proxy do RandomForest
    (em produção, carrega o modelo do MLflow Registry).

    Args:
        request: PredictRequest com transaction_id e features.

    Returns:
        PredictResponse com probabilidade e decisão APPROVED/BLOCKED.
    """
    v14 = request.features.get("V14", 0.0)
    amount = request.features.get("Amount", 0.0)

    fraud_prob = max(0.0, min(1.0, 0.5 - v14 * 0.05 + amount / 100_000))
    decision = "BLOCKED" if fraud_prob >= 0.35 else "APPROVED"

    logger.info(
        "Predição realizada",
        extra={
            "tx": request.transaction_id,
            "prob": round(fraud_prob, 4),
            "decision": decision,
        },
    )
    return PredictResponse(
        transaction_id=request.transaction_id,
        fraud_probability=round(fraud_prob, 4),
        decision=decision,
        model_version="1.0.0",
    )


@app.post("/ask", response_model=AskResponse, tags=["agent"])
async def ask(request: AskRequest) -> AskResponse:
    """Processa uma pergunta via agente ReAct de saúde bancária.

    Aplica guardrail de input antes de delegar ao agente.
    O agente usa ferramentas (FraudMetrics, DriftChecker, etc.) para
    compor uma resposta fundamentada com raciocínio Thought/Action/Observation.

    Args:
        request: AskRequest com query do usuário.

    Returns:
        AskResponse com resposta sanitizada, ferramentas usadas e iterações.

    Raises:
        HTTPException 400: Se a query for bloqueada pelo guardrail (prompt injection).
        HTTPException 503: Se o agente não estiver inicializado.
    """
    guardrail_result = _input_guardrail.check(request.query)
    if not guardrail_result["allowed"]:
        logger.warning(
            "Query bloqueada no endpoint /ask",
            extra={
                "reason": guardrail_result["reason"],
                "session_id": request.session_id,
            },
        )
        raise HTTPException(
            status_code=400,
            detail=f"Requisição bloqueada: {guardrail_result['reason']} (OWASP {guardrail_result['category']})",
        )

    if _agent is None:
        logger.error("Agente não disponível — ANTHROPIC_API_KEY configurada?")
        raise HTTPException(
            status_code=503,
            detail="Agente não disponível. Configure ANTHROPIC_API_KEY.",
        )

    try:
        result = _agent.ask(request.query)
        logger.info(
            "Resposta do agente gerada",
            extra={
                "session_id": request.session_id,
                "iterations": result["iterations"],
                "tools": result["tools_used"],
                "had_pii": result["had_pii"],
            },
        )
        return AskResponse(
            answer=result["answer"],
            tools_used=result["tools_used"],
            iterations=result["iterations"],
            had_pii=result["had_pii"],
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.error("Erro no agente", extra={"error": str(exc)})
        raise HTTPException(status_code=500, detail="Erro interno no agente.") from exc


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    uvicorn.run(app, host="0.0.0.0", port=8000, log_config=None)  # nosec B104
