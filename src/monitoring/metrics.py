"""Exportador de métricas Prometheus para observabilidade do sistema."""

from __future__ import annotations

import logging

from prometheus_client import Counter, Gauge, Histogram, start_http_server

logger = logging.getLogger(__name__)

REQUEST_LATENCY = Histogram(
    "request_latency_seconds",
    "Latência de requests por endpoint",
    ["endpoint", "method"],
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0],
)

REQUEST_TOTAL = Counter(
    "request_total",
    "Total de requests por endpoint e status",
    ["endpoint", "status_code"],
)

RAGAS_FAITHFULNESS = Gauge(
    "ragas_faithfulness_score",
    "Score de faithfulness RAGAS (rolling 24h)",
)

RAGAS_RELEVANCY = Gauge(
    "ragas_answer_relevancy_score",
    "Score de answer relevancy RAGAS (rolling 24h)",
)

DRIFT_PSI_SCORE = Gauge(
    "drift_psi_score",
    "PSI de drift por feature",
    ["feature"],
)

MODEL_AUC_CURRENT = Gauge(
    "model_auc_current",
    "AUC-ROC atual do modelo champion em produção",
)

FRAUD_RATE_ROLLING = Gauge(
    "fraud_rate_rolling_1h",
    "Taxa de fraude nas últimas 1h (transações bloqueadas / total)",
)

LLM_TOKENS_USED = Counter(
    "llm_tokens_used_total",
    "Total de tokens consumidos por modelo LLM",
    ["model", "direction"],  # direction: input | output
)

GUARDRAIL_BLOCKS = Counter(
    "guardrail_blocks_total",
    "Total de requisições bloqueadas pelos guardrails",
    ["category"],  # LLM01, LLM04, etc.
)

PII_DETECTIONS = Counter(
    "pii_detections_total",
    "Total de detecções de PII no output sanitizadas",
    ["pii_type"],
)


def update_drift_metrics(psi_by_feature: dict[str, float]) -> None:
    """Atualiza métricas de drift no Prometheus.

    Args:
        psi_by_feature: Dicionário mapeando feature -> PSI score.
    """
    for feature, psi in psi_by_feature.items():
        DRIFT_PSI_SCORE.labels(feature=feature).set(psi)
    logger.info("Métricas de drift atualizadas", extra={"features": len(psi_by_feature)})


def update_ragas_metrics(faithfulness: float, relevancy: float) -> None:
    """Atualiza métricas RAGAS no Prometheus.

    Args:
        faithfulness: Score de faithfulness RAGAS (0-1).
        relevancy: Score de answer relevancy RAGAS (0-1).
    """
    RAGAS_FAITHFULNESS.set(faithfulness)
    RAGAS_RELEVANCY.set(relevancy)
    logger.info("Métricas RAGAS atualizadas", extra={"faithfulness": faithfulness, "relevancy": relevancy})


def record_llm_tokens(model: str, input_tokens: int, output_tokens: int) -> None:
    """Registra tokens consumidos por uma chamada ao LLM.

    Args:
        model: Identificador do modelo LLM.
        input_tokens: Tokens no prompt de entrada.
        output_tokens: Tokens na resposta gerada.
    """
    LLM_TOKENS_USED.labels(model=model, direction="input").inc(input_tokens)
    LLM_TOKENS_USED.labels(model=model, direction="output").inc(output_tokens)


def start_metrics_server(port: int = 9090) -> None:
    """Inicia o servidor HTTP de métricas Prometheus em porta dedicada.

    Args:
        port: Porta do servidor de métricas. Default: 9090.
    """
    start_http_server(port)
    logger.info("Servidor de métricas Prometheus iniciado", extra={"port": port})
