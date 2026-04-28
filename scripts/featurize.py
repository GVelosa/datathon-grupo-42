"""DVC featurize stage: carrega dados brutos, computa features e faz upsert no Feature Store."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.features.feature_engineering import compute_features, load_raw, upsert_feature_store

df = load_raw("data/raw/creditcard.csv")
features = compute_features(df)
n = upsert_feature_store(features, "data/processed/features.parquet")
print(f"Upsert concluido: {n} registros no Feature Store")
