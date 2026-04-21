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
