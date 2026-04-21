"""Detecção de data drift e prediction drift com Evidently e PSI."""

from __future__ import annotations

import logging
from typing import Literal, TypedDict

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

PSI_THRESHOLD_WARNING: float = 0.20
PSI_THRESHOLD_CRITICAL: float = 0.25
N_BINS: int = 10


class DriftAlert(TypedDict):
    """Alerta de drift para uma feature.

    Attributes:
        feature: Nome da feature com drift detectado.
        psi_score: PSI calculado.
        severity: Nível de severidade do alerta.
        threshold: Threshold que foi excedido.
    """

    feature: str
    psi_score: float
    severity: Literal["WARNING", "CRITICAL"]
    threshold: float


class DriftReport(TypedDict):
    """Relatório consolidado de drift.

    Attributes:
        has_drift: Se pelo menos um alerta foi gerado.
        alerts: Lista de alertas por feature.
        overall_status: Status agregado do sistema.
        features_checked: Número de features verificadas.
        psi_by_feature: Dicionário com PSI por feature.
    """

    has_drift: bool
    alerts: list[DriftAlert]
    overall_status: Literal["OK", "WARNING", "CRITICAL"]
    features_checked: int
    psi_by_feature: dict[str, float]


def compute_psi(reference: np.ndarray, current: np.ndarray, n_bins: int = N_BINS) -> float:
    """Calcula Population Stability Index (PSI) entre duas distribuições.

    PSI < 0.10: Sem mudança
    PSI 0.10-0.20: Mudança moderada
    PSI > 0.20: Mudança significativa (WARNING)
    PSI > 0.25: Mudança crítica (CRITICAL)

    Args:
        reference: Distribuição de referência (dados de treino).
        current: Distribuição atual (dados de produção).
        n_bins: Número de bins para discretização. Default: 10.

    Returns:
        Valor PSI. Quanto maior, maior a diferença entre distribuições.
    """
    bins = np.percentile(reference, np.linspace(0, 100, n_bins + 1))
    bins[0] = -np.inf
    bins[-1] = np.inf

    ref_counts, _ = np.histogram(reference, bins=bins)
    cur_counts, _ = np.histogram(current, bins=bins)

    # Evitar divisão por zero com smoothing mínimo
    ref_pct = (ref_counts + 1e-6) / (len(reference) + n_bins * 1e-6)
    cur_pct = (cur_counts + 1e-6) / (len(current) + n_bins * 1e-6)

    psi = float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))
    return psi


def compute_prediction_drift(
    reference_scores: np.ndarray,
    current_scores: np.ndarray,
) -> tuple[float, bool]:
    """Verifica drift nas predições via teste de Kolmogorov-Smirnov.

    Args:
        reference_scores: Scores de probabilidade do período de referência.
        current_scores: Scores de probabilidade do período atual.

    Returns:
        Tupla (ks_statistic, has_drift) onde has_drift = True se p-value < 0.05.
    """
    from scipy.stats import ks_2samp

    ks_stat, p_value = ks_2samp(reference_scores, current_scores)
    has_drift = p_value < 0.05
    logger.info(
        "Prediction drift verificado",
        extra={"ks_statistic": round(ks_stat, 4), "p_value": round(p_value, 4), "has_drift": has_drift},
    )
    return float(ks_stat), has_drift


def check_all_features(
    df_reference: pd.DataFrame,
    df_current: pd.DataFrame,
    features: list[str] | None = None,
) -> DriftReport:
    """Verifica drift para todas as features especificadas.

    Args:
        df_reference: DataFrame de referência (dados de treino ou baseline).
        df_current: DataFrame atual (dados de produção recentes).
        features: Lista de features a verificar. Se None, usa interseção das colunas numéricas.

    Returns:
        DriftReport com status, alertas e PSI por feature.
    """
    if features is None:
        features = list(df_reference.select_dtypes(include=[np.number]).columns)

    alerts: list[DriftAlert] = []
    psi_by_feature: dict[str, float] = {}

    for feature in features:
        if feature not in df_reference.columns or feature not in df_current.columns:
            logger.warning("Feature ausente em um dos datasets", extra={"feature": feature})
            continue

        ref_vals = df_reference[feature].dropna().values
        cur_vals = df_current[feature].dropna().values
        psi = compute_psi(ref_vals, cur_vals)
        psi_by_feature[feature] = round(psi, 4)

        if psi >= PSI_THRESHOLD_CRITICAL:
            alerts.append(DriftAlert(
                feature=feature,
                psi_score=psi,
                severity="CRITICAL",
                threshold=PSI_THRESHOLD_CRITICAL,
            ))
        elif psi >= PSI_THRESHOLD_WARNING:
            alerts.append(DriftAlert(
                feature=feature,
                psi_score=psi,
                severity="WARNING",
                threshold=PSI_THRESHOLD_WARNING,
            ))

    has_drift = len(alerts) > 0
    has_critical = any(a["severity"] == "CRITICAL" for a in alerts)
    overall_status: Literal["OK", "WARNING", "CRITICAL"] = (
        "CRITICAL" if has_critical else ("WARNING" if has_drift else "OK")
    )

    logger.info(
        "Drift check concluído",
        extra={"features_checked": len(psi_by_feature), "alerts": len(alerts), "status": overall_status},
    )
    return DriftReport(
        has_drift=has_drift,
        alerts=alerts,
        overall_status=overall_status,
        features_checked=len(psi_by_feature),
        psi_by_feature=psi_by_feature,
    )
