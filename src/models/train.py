"""Orquestrador de treino com MLflow tracking e gate de promoção Champion-Challenger."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.features.feature_engineering import FEATURE_COLS, TARGET_COL, get_train_val_test_splits
from src.models.baseline import FraudRandomForest

logger = logging.getLogger(__name__)

PROMOTION_DELTA_AUC: float = 0.005
MLFLOW_EXPERIMENT_NAME: str = "fraud-detection"


def compute_metrics(y_true: np.ndarray, y_proba: np.ndarray, threshold: float = 0.35) -> dict[str, float]:
    """Calcula métricas de classificação para detecção de fraude.

    Args:
        y_true: Labels verdadeiros com shape (n_samples,).
        y_proba: Probabilidades preditas com shape (n_samples,).
        threshold: Threshold de decisão binária.

    Returns:
        Dicionário com métricas: auc_roc, average_precision, f1, precision, recall, ks_statistic.
    """
    y_pred = (y_proba >= threshold).astype(int)

    # KS Statistic: separação máxima entre distribuições de positivos e negativos
    pos_proba = y_proba[y_true == 1]
    neg_proba = y_proba[y_true == 0]
    ks = float(np.max(np.abs(
        np.searchsorted(np.sort(pos_proba), y_proba) / len(pos_proba)
        - np.searchsorted(np.sort(neg_proba), y_proba) / len(neg_proba)
    ))) if len(pos_proba) > 0 else 0.0

    return {
        "auc_roc": float(roc_auc_score(y_true, y_proba)),
        "average_precision": float(average_precision_score(y_true, y_proba)),
        "f1_score": float(f1_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred)),
        "recall": float(recall_score(y_true, y_pred)),
        "ks_statistic": ks,
    }


def train_and_log(
    df_features: pd.DataFrame,
    config: dict[str, Any] | None = None,
    experiment_name: str = MLFLOW_EXPERIMENT_NAME,
) -> tuple[str, dict[str, float]]:
    """Treina o modelo baseline e loga tudo no MLflow.

    Args:
        df_features: DataFrame com features e target.
        config: Hiperparâmetros do modelo. Usa defaults se None.
        experiment_name: Nome do experimento MLflow.

    Returns:
        Tupla (run_id, metrics) com o ID do run e as métricas de validação.
    """
    cfg = config or {}
    mlflow.set_experiment(experiment_name)

    df_train, df_val, _ = get_train_val_test_splits(df_features)

    X_train = df_train[FEATURE_COLS].values
    y_train = df_train[TARGET_COL].values
    X_val = df_val[FEATURE_COLS].values
    y_val = df_val[TARGET_COL].values

    with mlflow.start_run() as run:
        model = FraudRandomForest(config=cfg)
        model.fit(X_train, y_train)

        mlflow.log_params({
            "model_type": "random_forest",
            "n_estimators": model.model.n_estimators,
            "max_depth": model.model.max_depth,
            "class_weight": str(model.model.class_weight),
            "decision_threshold": model.threshold,
            "n_features": len(FEATURE_COLS),
        })

        val_proba = model.predict_proba(X_val)
        metrics = compute_metrics(y_val, val_proba, threshold=model.threshold)
        mlflow.log_metrics(metrics)

        mlflow.set_tags({
            "champion": "false",
            "environment": "staging",
            "n_train_samples": str(len(X_train)),
            "n_val_samples": str(len(X_val)),
        })

        mlflow.sklearn.log_model(model.model, artifact_path="model")
        logger.info("Run MLflow criado", extra={"run_id": run.info.run_id, "auc": metrics["auc_roc"]})

        return run.info.run_id, metrics


def evaluate_champion_challenger(
    challenger_auc: float,
    champion_run_id: str | None,
    client: mlflow.MlflowClient | None = None,
) -> bool:
    """Verifica se o challenger supera o champion pelo delta mínimo de AUC.

    Args:
        challenger_auc: AUC-ROC do modelo challenger.
        champion_run_id: Run ID do modelo champion atual. None se não houver champion.
        client: Cliente MLflow. Se None, cria um novo.

    Returns:
        True se o challenger deve ser promovido, False caso contrário.
    """
    if champion_run_id is None:
        logger.info("Sem champion — challenger promovido automaticamente")
        return True

    if client is None:
        client = mlflow.MlflowClient()

    champion_run = client.get_run(champion_run_id)
    champion_auc = float(champion_run.data.metrics.get("auc_roc", 0.0))
    delta = challenger_auc - champion_auc

    logger.info(
        "Champion-Challenger avaliado",
        extra={"challenger_auc": challenger_auc, "champion_auc": champion_auc, "delta": delta},
    )
    return delta >= PROMOTION_DELTA_AUC
