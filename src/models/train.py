"""Orquestrador de treino com MLflow tracking e gate de promoção Champion-Challenger."""

from __future__ import annotations

import json
import logging
import tempfile
from pathlib import Path
from typing import Any

import mlflow
import mlflow.pytorch
import mlflow.sklearn
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    fbeta_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.features.feature_engineering import FEATURE_COLS, TARGET_COL, get_train_val_test_splits
from src.models.baseline import FraudRandomForest, MLPFraudDetector, build_mlp_trainer

logger = logging.getLogger(__name__)

PROMOTION_DELTA: float = 0.005
PROMOTION_PRIMARY_METRIC: str = "average_precision"
MLFLOW_EXPERIMENT_NAME: str = "fraud-detection"


def compute_metrics(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    threshold: float | None = None,
) -> dict[str, float]:
    """Calcula métricas de classificação para detecção de fraude.

    Quando threshold é None, calcula o threshold ótimo que maximiza F2-score
    na curva Precision-Recall (penaliza mais falsos negativos — fraude não detectada
    é mais custosa que falso alarme em contexto bancário).

    Args:
        y_true: Labels verdadeiros com shape (n_samples,).
        y_proba: Probabilidades preditas com shape (n_samples,).
        threshold: Threshold de decisão binária. Se None, usa threshold ótimo via F2.

    Returns:
        Dicionário com métricas: auc_roc, average_precision, f1_score, f2_score,
        precision, recall, ks_statistic, optimal_threshold, effective_threshold.
    """
    # Threshold ótimo via curva PR maximizando F2-score (beta=2 → recall 2x mais importante)
    precisions, recalls, pthresholds = precision_recall_curve(y_true, y_proba)
    f2_arr = (5 * precisions[:-1] * recalls[:-1]) / (4 * precisions[:-1] + recalls[:-1] + 1e-9)
    optimal_threshold = float(pthresholds[f2_arr.argmax()]) if len(pthresholds) > 0 else 0.35

    effective_threshold = threshold if threshold is not None else optimal_threshold
    y_pred = (y_proba >= effective_threshold).astype(int)

    pos_proba = y_proba[y_true == 1]
    neg_proba = y_proba[y_true == 0]
    if len(pos_proba) > 0 and len(neg_proba) > 0:
        pos_sorted = np.sort(pos_proba)
        neg_sorted = np.sort(neg_proba)
        ks = float(
            np.max(
                np.abs(
                    np.searchsorted(pos_sorted, y_proba) / len(pos_proba)
                    - np.searchsorted(neg_sorted, y_proba) / len(neg_proba)
                )
            )
        )
    else:
        ks = 0.0

    return {
        "auc_roc": float(roc_auc_score(y_true, y_proba)),
        "average_precision": float(average_precision_score(y_true, y_proba)),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
        "f2_score": float(fbeta_score(y_true, y_pred, beta=2, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "ks_statistic": ks,
        "optimal_threshold": optimal_threshold,
        "effective_threshold": effective_threshold,
    }


def _log_shap_artifact(
    model: FraudRandomForest,
    X_val: np.ndarray,
    feature_names: list[str],
) -> None:
    """Calcula SHAP values e loga como artefato MLflow.

    Args:
        model: Modelo RandomForest treinado com scaler.
        X_val: Features de validação (não normalizadas) com shape (n_samples, n_features).
        feature_names: Lista com nomes das features na ordem de X_val.
    """
    try:
        import shap

        X_val_scaled = model.scaler.transform(X_val)
        explainer = shap.TreeExplainer(model.model)
        shap_sample = X_val_scaled[:500]
        shap_values = explainer.shap_values(shap_sample)

        if isinstance(shap_values, list):
            shap_fraud = shap_values[1]
        else:
            shap_fraud = shap_values

        mean_abs_shap = np.abs(shap_fraud).mean(axis=0)
        shap_importance = pd.DataFrame(
            {
                "feature": feature_names,
                "mean_abs_shap": mean_abs_shap,
            }
        ).sort_values("mean_abs_shap", ascending=False)

        with tempfile.TemporaryDirectory() as tmpdir:
            shap_path = Path(tmpdir) / "shap_importance.csv"
            shap_importance.to_csv(shap_path, index=False)
            mlflow.log_artifact(str(shap_path), artifact_path="shap")

            shap_json_path = Path(tmpdir) / "shap_summary.json"
            shap_json_path.write_text(
                json.dumps(
                    {
                        "top_features": shap_importance.head(10)[["feature", "mean_abs_shap"]]
                        .assign(mean_abs_shap=lambda df: df["mean_abs_shap"].round(6))
                        .to_dict("records"),
                        "n_samples_explained": len(shap_sample),
                    },
                    indent=2,
                )
            )
            mlflow.log_artifact(str(shap_json_path), artifact_path="shap")

            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(10, 8))
            top_n = min(15, len(shap_importance))
            top_features = shap_importance.head(top_n)
            ax.barh(
                top_features["feature"][::-1], top_features["mean_abs_shap"][::-1], color="#4878CF", edgecolor="white"
            )
            ax.set_xlabel("Mean |SHAP value|")
            ax.set_title(f"SHAP Feature Importance — Top {top_n} Features", fontweight="bold")
            plt.tight_layout()
            shap_plot_path = Path(tmpdir) / "shap_barplot.png"
            fig.savefig(shap_plot_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            mlflow.log_artifact(str(shap_plot_path), artifact_path="shap")

        logger.info("SHAP artifacts logados", extra={"top_feature": shap_importance.iloc[0]["feature"]})

    except ImportError:
        logger.warning("shap não instalado — artifact SHAP ignorado")
    except Exception as exc:
        logger.warning("Falha ao gerar SHAP artifacts", extra={"error": str(exc)})


def train_and_log(
    df_features: pd.DataFrame,
    config: dict[str, Any] | None = None,
    experiment_name: str = MLFLOW_EXPERIMENT_NAME,
) -> tuple[str, dict[str, float]]:
    """Treina o modelo RandomForest baseline e loga tudo no MLflow.

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

    X_train = df_train[FEATURE_COLS].values.astype(np.float32)
    y_train = df_train[TARGET_COL].values.astype(np.int32)
    X_val = df_val[FEATURE_COLS].values.astype(np.float32)
    y_val = df_val[TARGET_COL].values.astype(np.int32)

    smote_strategy: float = float(cfg.get("smote_sampling_strategy", 0.0))
    smote_applied = False
    if smote_strategy > 0.0:
        try:
            from imblearn.over_sampling import SMOTE

            sm = SMOTE(random_state=int(cfg.get("random_state", 42)), sampling_strategy=smote_strategy)
            X_train, y_train = sm.fit_resample(X_train, y_train)
            smote_applied = True
            logger.info(
                "SMOTE aplicado",
                extra={"strategy": smote_strategy, "n_after": len(X_train), "fraud_after": int(y_train.sum())},
            )
        except ImportError:
            logger.warning("imbalanced-learn não instalado — SMOTE ignorado. pip install imbalanced-learn")

    with mlflow.start_run(run_name="random_forest_baseline") as run:
        model = FraudRandomForest(config=cfg)
        model.fit(X_train, y_train)

        mlflow.log_params(
            {
                "model_type": "random_forest",
                "n_estimators": model.model.n_estimators,
                "max_depth": model.model.max_depth,
                "class_weight": str(model.model.class_weight),
                "min_samples_leaf": model.model.min_samples_leaf,
                "n_features": len(FEATURE_COLS),
                "n_train_samples": len(X_train),
                "n_val_samples": len(X_val),
                "smote_applied": smote_applied,
                "smote_strategy": smote_strategy,
            }
        )

        val_proba = model.predict_proba(X_val)
        metrics = compute_metrics(y_val, val_proba)  # threshold dinâmico via F2
        mlflow.log_metrics(metrics)

        mlflow.set_tags(
            {
                "champion": "false",
                "environment": "staging",
                "model_family": "tree_ensemble",
            }
        )

        mlflow.sklearn.log_model(model.model, artifact_path="model")

        _log_shap_artifact(model, X_val, FEATURE_COLS)

        logger.info(
            "Run MLflow (RF) criado",
            extra={
                "run_id": run.info.run_id,
                "pr_auc": metrics["average_precision"],
                "auc_roc": metrics["auc_roc"],
                "f2_score": metrics["f2_score"],
                "optimal_threshold": metrics["optimal_threshold"],
            },
        )
        return run.info.run_id, metrics


def train_mlp_and_log(
    df_features: pd.DataFrame,
    config: dict[str, Any] | None = None,
    experiment_name: str = MLFLOW_EXPERIMENT_NAME,
) -> tuple[str, dict[str, float]]:
    """Treina o modelo MLP (PyTorch) e loga tudo no MLflow.

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

    X_train_np = df_train[FEATURE_COLS].values.astype(np.float32)
    y_train_np = df_train[TARGET_COL].values.astype(np.float32)

    smote_strategy_mlp: float = float(cfg.get("smote_sampling_strategy", 0.0))
    smote_applied_mlp = False
    if smote_strategy_mlp > 0.0:
        try:
            from imblearn.over_sampling import SMOTE

            sm = SMOTE(random_state=int(cfg.get("random_state", 42)), sampling_strategy=smote_strategy_mlp)
            X_train_np, y_train_np = sm.fit_resample(X_train_np, y_train_np)
            smote_applied_mlp = True
            logger.info(
                "SMOTE aplicado (MLP)",
                extra={"strategy": smote_strategy_mlp, "n_after": len(X_train_np)},
            )
        except ImportError:
            logger.warning("imbalanced-learn não instalado — SMOTE ignorado no MLP")

    X_train = torch.tensor(X_train_np.astype(np.float32), dtype=torch.float32)
    y_train = torch.tensor(y_train_np.astype(np.float32), dtype=torch.float32)
    X_val_tensor = torch.tensor(df_val[FEATURE_COLS].values.astype(np.float32), dtype=torch.float32)
    y_val_np = df_val[TARGET_COL].values.astype(np.float32)

    input_dim = len(FEATURE_COLS)
    hidden_layers: list[int] = cfg.get("hidden_layers", [128, 64, 32])
    dropout: float = float(cfg.get("dropout", 0.3))
    pos_weight: float = float(cfg.get("pos_weight", 577.0))
    lr: float = float(cfg.get("learning_rate", 1e-3))
    epochs: int = int(cfg.get("epochs", 50))
    batch_size: int = int(cfg.get("batch_size", 1024))
    patience: int = int(cfg.get("early_stopping_patience", 5))

    with mlflow.start_run(run_name="mlp_baseline") as run:
        mlflow.log_params(
            {
                "model_type": "mlp_pytorch",
                "input_dim": input_dim,
                "hidden_layers": str(hidden_layers),
                "dropout": dropout,
                "pos_weight": pos_weight,
                "learning_rate": lr,
                "epochs_max": epochs,
                "batch_size": batch_size,
                "early_stopping_patience": patience,
                "n_features": input_dim,
                "n_train_samples": len(X_train),
                "n_val_samples": len(X_val_tensor),
                "smote_applied": smote_applied_mlp,
                "smote_strategy": smote_strategy_mlp,
            }
        )

        model = MLPFraudDetector(input_dim=input_dim, hidden_layers=hidden_layers, dropout=dropout)
        criterion, optimizer = build_mlp_trainer(model, pos_weight=pos_weight, lr=lr)

        dataset = torch.utils.data.TensorDataset(X_train, y_train)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        best_val_loss = float("inf")
        no_improve_count = 0
        best_epoch = 0

        for epoch in range(epochs):
            model.train()
            epoch_loss = 0.0
            for X_batch, y_batch in loader:
                optimizer.zero_grad()
                logits = model(X_batch).squeeze(-1)
                loss = criterion(logits, y_batch)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * len(X_batch)

            avg_loss = epoch_loss / len(X_train)

            model.eval()
            with torch.no_grad():
                val_logits = model(X_val_tensor).squeeze(-1)
                val_loss = criterion(val_logits, torch.tensor(y_val_np, dtype=torch.float32)).item()

            mlflow.log_metrics({"train_loss": avg_loss, "val_loss": val_loss}, step=epoch)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                no_improve_count = 0
            else:
                no_improve_count += 1
                if no_improve_count >= patience:
                    logger.info(
                        "Early stopping ativado",
                        extra={"epoch": epoch, "best_epoch": best_epoch},
                    )
                    break

        val_proba = model.predict_proba(X_val_tensor).numpy()
        metrics = compute_metrics(y_val_np, val_proba)  # threshold dinâmico via F2
        mlflow.log_metrics(metrics)
        mlflow.log_metric("best_epoch", float(best_epoch))

        mlflow.set_tags(
            {
                "champion": "false",
                "environment": "staging",
                "model_family": "neural_network",
                "framework": "pytorch",
            }
        )

        mlflow.pytorch.log_model(model, artifact_path="model")

        logger.info(
            "Run MLflow (MLP) criado",
            extra={
                "run_id": run.info.run_id,
                "pr_auc": metrics["average_precision"],
                "auc_roc": metrics["auc_roc"],
                "f2_score": metrics["f2_score"],
                "optimal_threshold": metrics["optimal_threshold"],
            },
        )
        return run.info.run_id, metrics


def evaluate_champion_challenger(
    challenger_metrics: dict[str, float],
    champion_run_id: str | None,
    client: mlflow.MlflowClient | None = None,
    primary_metric: str = PROMOTION_PRIMARY_METRIC,
    min_delta: float = PROMOTION_DELTA,
) -> bool:
    """Verifica se o challenger supera o champion na métrica primária.

    Usa PR-AUC (average_precision) como métrica primária por ser mais informativa
    que AUC-ROC em datasets desbalanceados (fraude ≈ 0.17% das transações).
    Fallback para auc_roc se primary_metric não estiver disponível no champion.

    Args:
        challenger_metrics: Métricas do modelo challenger (saída de compute_metrics).
        champion_run_id: Run ID do modelo champion atual. None se não houver champion.
        client: Cliente MLflow. Se None, cria um novo.
        primary_metric: Métrica de comparação. Default: average_precision (PR-AUC).
        min_delta: Delta mínimo necessário para promoção.

    Returns:
        True se o challenger deve ser promovido, False caso contrário.
    """
    challenger_score = challenger_metrics.get(primary_metric, challenger_metrics.get("auc_roc", 0.0))

    if champion_run_id is None:
        logger.info("Sem champion — challenger promovido automaticamente")
        return True

    if client is None:
        client = mlflow.MlflowClient()

    champion_run = client.get_run(champion_run_id)
    champion_score = float(champion_run.data.metrics.get(primary_metric, champion_run.data.metrics.get("auc_roc", 0.0)))
    delta = challenger_score - champion_score

    logger.info(
        "Champion-Challenger avaliado",
        extra={
            "primary_metric": primary_metric,
            "challenger_score": challenger_score,
            "champion_score": champion_score,
            "delta": delta,
            "gate": min_delta,
            "promoted": delta >= min_delta,
        },
    )
    return delta >= min_delta


if __name__ == "__main__":
    import yaml
    from dotenv import load_dotenv

    load_dotenv()  # carrega .env da raiz para garantir MLFLOW_TRACKING_URI, etc.

    from src.features.feature_engineering import compute_features, load_raw, upsert_feature_store

    logging.basicConfig(level=logging.INFO)

    config_path = Path("configs/model_config.yaml")
    with config_path.open() as f:
        full_config = yaml.safe_load(f)

    smote_config = full_config.get("smote", {})
    smote_strategy_cfg = smote_config.get("sampling_strategy", 0.0) if smote_config.get("enabled", False) else 0.0
    train_config = full_config.get("training", {})

    rf_config = {**full_config.get("random_forest", {}), "smote_sampling_strategy": smote_strategy_cfg}
    mlp_config = {**full_config.get("mlp", {}), "smote_sampling_strategy": smote_strategy_cfg}

    raw_path = Path("data/raw/creditcard.csv")
    feature_store_path = Path("data/processed/features.parquet")

    df_raw = load_raw(raw_path)
    df_features = compute_features(df_raw)
    upsert_feature_store(df_features, feature_store_path)
    df_fs = pd.read_parquet(feature_store_path)

    experiment_name: str = train_config.get("experiment_name", MLFLOW_EXPERIMENT_NAME)

    rf_run_id, rf_metrics = train_and_log(df_fs, config=rf_config, experiment_name=experiment_name)
    mlp_run_id, mlp_metrics = train_mlp_and_log(df_fs, config=mlp_config, experiment_name=experiment_name)

    best_pr_auc = max(rf_metrics["average_precision"], mlp_metrics["average_precision"])
    best_run = rf_run_id if rf_metrics["average_precision"] >= mlp_metrics["average_precision"] else mlp_run_id

    eval_output = {
        "rf_run_id": rf_run_id,
        "rf_metrics": rf_metrics,
        "mlp_run_id": mlp_run_id,
        "mlp_metrics": mlp_metrics,
        "best_run_id": best_run,
        "best_pr_auc": best_pr_auc,
        "primary_metric": PROMOTION_PRIMARY_METRIC,
        "promotion_gate": PROMOTION_DELTA,
    }

    eval_path = Path("data/processed/eval_metrics.json")
    eval_path.parent.mkdir(parents=True, exist_ok=True)
    eval_path.write_text(json.dumps(eval_output, indent=2))

    logger.info(
        "Treino completo",
        extra={
            "rf_pr_auc": rf_metrics["average_precision"],
            "mlp_pr_auc": mlp_metrics["average_precision"],
            "best_run": best_run,
            "best_pr_auc": best_pr_auc,
        },
    )
