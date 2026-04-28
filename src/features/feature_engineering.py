"""Feature engineering pipeline com upsert incremental no Feature Store."""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

FEATURE_COLS: list[str] = [f"V{i}" for i in range(1, 29)] + [
    "Amount",
    "hour_of_day",
    "is_night",
    "log_amount",
    "amount_zscore",
]
# 33 features total: V1-V28 (PCA) + Amount + 4 derived (hour_of_day, is_night, log_amount, amount_zscore)
TARGET_COL: str = "Class"


def load_raw(path: str | Path) -> pd.DataFrame:
    """Carrega dados brutos do CSV de transações.

    Args:
        path: Caminho para o arquivo CSV de transações brutas.

    Returns:
        DataFrame com colunas originais: Time, V1-V28, Amount, Class.

    Raises:
        FileNotFoundError: Se o arquivo não existir no path informado.
        ValueError: Se colunas obrigatórias estiverem ausentes.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset não encontrado: {path}")

    df = pd.read_csv(path)

    required_cols = {"Time", "Amount", "Class"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Colunas obrigatórias ausentes: {missing}")

    logger.info("Dataset carregado", extra={"path": str(path), "rows": len(df)})
    return df


def compute_features(df: pd.DataFrame, reference_date: datetime | None = None) -> pd.DataFrame:
    """Calcula features derivadas a partir das transações brutas.

    Args:
        df: DataFrame de entrada com colunas Time, Amount, V1-V28, Class.
        reference_date: Data de referência para cálculos. Se None, usa datetime.now().

    Returns:
        DataFrame com features adicionais: hour_of_day, is_night, log_amount, amount_zscore.
    """
    if reference_date is None:
        reference_date = datetime.now()

    result = df.copy()
    result["hour_of_day"] = (result["Time"] % 86400 / 3600).astype(int)
    result["is_night"] = result["hour_of_day"].between(22, 23) | result["hour_of_day"].between(0, 5)
    result["log_amount"] = np.log1p(result["Amount"])
    mean_amount = result["Amount"].mean()
    std_amount = result["Amount"].std()
    result["amount_zscore"] = (result["Amount"] - mean_amount) / (std_amount + 1e-8)
    result["feature_version"] = reference_date.strftime("%Y%m%d_%H%M%S")

    logger.info("Features computadas", extra={"rows": len(result)})
    return result


def upsert_feature_store(
    features: pd.DataFrame,
    table_path: str | Path,
    primary_key: str = "index",
) -> int:
    """Realiza upsert incremental no Feature Store (nunca flush destrutivo).

    Args:
        features: DataFrame com features calculadas.
        table_path: Caminho do arquivo Parquet do Feature Store.
        primary_key: Coluna usada como chave de merge.

    Returns:
        Número de registros afetados pelo upsert.

    Raises:
        ValueError: Se primary_key não estiver presente em features.
    """
    table_path = Path(table_path)

    if primary_key != "index" and primary_key not in features.columns:
        raise ValueError(f"primary_key '{primary_key}' não encontrado em features.columns")

    if primary_key == "index":
        features = features.reset_index()

    if table_path.exists():
        existing = pd.read_parquet(table_path)
        merged = existing.set_index(primary_key)
        incoming = features.set_index(primary_key)
        merged.update(incoming)
        new_keys = incoming.index.difference(merged.index)
        if len(new_keys) > 0:
            merged = pd.concat([merged, incoming.loc[new_keys]])
        affected = len(incoming)
        result = merged.reset_index()
    else:
        table_path.parent.mkdir(parents=True, exist_ok=True)
        result = features
        affected = len(features)

    result.to_parquet(table_path, index=False)
    logger.info("Upsert concluído", extra={"affected": affected, "total_rows": len(result)})
    return affected


def get_train_val_test_splits(
    features: pd.DataFrame,
    target_col: str = TARGET_COL,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Divide o dataset em treino, validação e teste com estratificação.

    Args:
        features: DataFrame completo com features e target.
        target_col: Nome da coluna target binária.
        train_ratio: Proporção para treino (default 0.70).
        val_ratio: Proporção para validação (default 0.15).
        random_state: Semente para reprodutibilidade.

    Returns:
        Tupla (df_train, df_val, df_test) com splits estratificados.
    """
    df_train, df_temp = train_test_split(
        features,
        test_size=(1 - train_ratio),
        stratify=features[target_col],
        random_state=random_state,
    )
    val_size_adjusted = val_ratio / (1 - train_ratio)
    df_val, df_test = train_test_split(
        df_temp,
        test_size=(1 - val_size_adjusted),
        stratify=df_temp[target_col],
        random_state=random_state,
    )
    logger.info("Splits criados", extra={"train": len(df_train), "val": len(df_val), "test": len(df_test)})
    return df_train, df_val, df_test
