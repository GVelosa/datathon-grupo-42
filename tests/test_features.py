import tempfile
from pathlib import Path

import pandas as pd
import pytest

from src.features.feature_engineering import (
    compute_features,
    get_train_val_test_splits,
    upsert_feature_store,
)


def test_compute_features_adds_columns(sample_transactions_df):
    result = compute_features(sample_transactions_df)
    assert "hour_of_day" in result.columns
    assert "is_night" in result.columns
    assert "log_amount" in result.columns
    assert "amount_zscore" in result.columns


def test_compute_features_hour_range(sample_transactions_df):
    result = compute_features(sample_transactions_df)
    assert result["hour_of_day"].between(0, 23).all()


def test_compute_features_log_amount_non_negative(sample_transactions_df):
    result = compute_features(sample_transactions_df)
    assert (result["log_amount"] >= 0).all()


def test_upsert_does_not_destroy_existing(sample_transactions_df):
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "feature_store.parquet"
        features = compute_features(sample_transactions_df)
        features = features.reset_index()
        upsert_feature_store(features.head(100), path, primary_key="index")
        upsert_feature_store(features.tail(50), path, primary_key="index")
        existing = pd.read_parquet(path)
        assert len(existing) >= 100, "Registros existentes nao devem ser deletados"


def test_upsert_creates_file_if_not_exists(sample_transactions_df):
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "new_store.parquet"
        assert not path.exists()
        features = compute_features(sample_transactions_df)
        upsert_feature_store(features, path)
        assert path.exists()


def test_upsert_raises_on_missing_primary_key(sample_transactions_df):
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "store.parquet"
        with pytest.raises(ValueError, match="primary_key"):
            upsert_feature_store(sample_transactions_df, path, primary_key="nonexistent_col")


def test_splits_are_stratified(sample_transactions_df):
    features = compute_features(sample_transactions_df)
    train, val, test = get_train_val_test_splits(features)
    total = len(train) + len(val) + len(test)
    assert total == len(features)
    assert len(train) > len(val)
    assert len(train) > len(test)
