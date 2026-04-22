"""Ferramentas do agente ReAct para consultas de saúde bancária."""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Literal, TypedDict

logger = logging.getLogger(__name__)


class FraudMetricsInput(TypedDict):
    """Input para FraudMetricsLookup."""
    period: str  # ex: "24h", "7d"


class FraudMetricsResult(TypedDict):
    """Resultado de métricas de fraude."""
    auc_roc: float
    f1_score: float
    fraud_rate: float
    total_transactions: int
    blocked_transactions: int
    last_updated: str


class DriftInput(TypedDict, total=False):
    """Input para DriftStatusChecker."""
    features: list[str] | None


class DriftAlert(TypedDict):
    """Alerta de drift individual."""
    feature: str
    psi_score: float
    severity: Literal["WARNING", "CRITICAL"]


class DriftResult(TypedDict):
    """Resultado do check de drift."""
    has_drift: bool
    alerts: list[DriftAlert]
    overall_status: Literal["OK", "WARNING", "CRITICAL"]
    checked_at: str


class FeatureInput(TypedDict):
    """Input para FeatureLookup."""
    transaction_id: str


class FeatureResult(TypedDict):
    """Resultado de lookup de features de uma transação."""
    transaction_id: str
    features: dict[str, float]
    fraud_probability: float
    decision: Literal["APPROVED", "BLOCKED", "REVIEW"]
    shap_top_features: list[tuple[str, float]]


class KBInput(TypedDict, total=False):
    """Input para KnowledgeBaseSearch."""
    query: str
    top_k: int


class KBChunk(TypedDict):
    """Chunk recuperado do knowledge base."""
    content: str
    source: str
    relevance_score: float


class KBResult(TypedDict):
    """Resultado de busca no knowledge base."""
    chunks: list[KBChunk]
    answer_context: str


class AlertInput(TypedDict, total=False):
    """Input para AlertHistoryQuery."""
    hours: int
    severity: str | None


class Alert(TypedDict):
    """Alerta individual do histórico."""
    type: str
    severity: Literal["INFO", "WARNING", "CRITICAL"]
    message: str
    timestamp: str


class AlertResult(TypedDict):
    """Resultado do histórico de alertas."""
    alerts: list[Alert]
    summary: str
    critical_count: int


def fraud_metrics_lookup(input_data: FraudMetricsInput) -> FraudMetricsResult:
    """Busca métricas atuais do modelo de fraude em produção.

    Args:
        input_data: Dicionário com período de consulta (ex: '24h', '7d').

    Returns:
        FraudMetricsResult com AUC, F1, taxa de fraude e volumes.
    """
    logger.info("FraudMetricsLookup chamado", extra={"period": input_data.get("period", "24h")})
    # Mock — em produção: consulta ao MLflow Registry + métricas do Prometheus
    return FraudMetricsResult(
        auc_roc=0.9743,
        f1_score=0.8821,
        fraud_rate=0.0023,
        total_transactions=142_891,
        blocked_transactions=329,
        last_updated=datetime.now().isoformat(),
    )


def drift_status_checker(input_data: DriftInput) -> DriftResult:
    """Verifica o status de drift dos dados e predições.

    Args:
        input_data: Opcionalmente filtra features específicas a verificar.

    Returns:
        DriftResult com alertas por feature e status geral.
    """
    logger.info("DriftStatusChecker chamado")
    # Mock — em produção: consulta ao DriftDetector com dados reais
    return DriftResult(
        has_drift=False,
        alerts=[],
        overall_status="OK",
        checked_at=datetime.now().isoformat(),
    )


def feature_lookup(input_data: FeatureInput) -> FeatureResult:
    """Busca features e decisão de uma transação específica por ID.

    Args:
        input_data: Dicionário com transaction_id a consultar.

    Returns:
        FeatureResult com features numéricas, probabilidade e SHAP top features.
    """
    tx_id = input_data["transaction_id"]
    logger.info("FeatureLookup chamado", extra={"tx_id": tx_id})
    # Mock — em produção: consulta ao Feature Store Delta Lake
    return FeatureResult(
        transaction_id=tx_id,
        features={"V14": -8.3, "V17": -2.1, "Amount": 4850.0, "hour_of_day": 2},
        fraud_probability=0.94,
        decision="BLOCKED",
        shap_top_features=[("V14", -0.42), ("Amount", 0.19), ("hour_of_day", 0.15)],
    )


def knowledge_base_search(input_data: KBInput) -> KBResult:
    """Busca informações no knowledge base interno via RAG (mock para dev).

    Args:
        input_data: Dicionário com query e opcionalmente top_k (padrão: 3).

    Returns:
        KBResult com lista de chunks relevantes e contexto concatenado.
    """
    query = input_data.get("query", "")
    top_k = input_data.get("top_k", 3) or 3

    logger.info("KnowledgeBaseSearch chamado", extra={"query_len": len(query), "top_k": top_k})

    mock_chunks: list[KBChunk] = [
        KBChunk(
            content=(
                "O modelo de fraude usa AUC-ROC como métrica principal. "
                "O threshold de decisão é 0.35 para maximizar recall. "
                "Base legal: LGPD Art. 7-IX (legítimo interesse), BACEN Resolução 4.658."
            ),
            source="MODEL_CARD.md",
            relevance_score=0.92,
        ),
        KBChunk(
            content=(
                "Drift detection usa PSI (Population Stability Index). "
                "Threshold de alerta: PSI > 0.2 = WARNING, PSI > 0.25 = CRITICAL. "
                "Monitoramento de janela 24h rolling vs. baseline de treino."
            ),
            source="SYSTEM_CARD.md",
            relevance_score=0.85,
        ),
        KBChunk(
            content=(
                "Features mais discriminativas: V14 (Cohen's d > 3), V17, V12. "
                "Transações com V14 < -5 têm probabilidade de fraude > 80%. "
                "SHAP values explicam decisões (Art. 20 LGPD — direito à explicação)."
            ),
            source="MODEL_CARD.md",
            relevance_score=0.78,
        ),
    ]

    context_parts = [f"[Fonte: {c['source']}]\n{c['content']}" for c in mock_chunks[:top_k]]
    answer_context = "\n\n---\n\n".join(context_parts)

    return KBResult(chunks=mock_chunks[:top_k], answer_context=answer_context)


def alert_history_query(input_data: AlertInput) -> AlertResult:
    """Busca histórico de alertas de fraude e drift.

    Args:
        input_data: Período em horas e opcionalmente filtro de severidade.

    Returns:
        AlertResult com lista de alertas, resumo e contagem de críticos.
    """
    hours = input_data.get("hours", 24)
    logger.info("AlertHistoryQuery chamado", extra={"hours": hours})
    # Mock — em produção: consulta ao Prometheus AlertManager
    return AlertResult(
        alerts=[],
        summary=f"Sem alertas críticos nas últimas {hours}h. Sistema operando normalmente.",
        critical_count=0,
    )
