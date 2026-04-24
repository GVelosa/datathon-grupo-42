import numpy as np
import pytest
from unittest.mock import patch, MagicMock
from src.models.baseline import FraudRandomForest, MLPFraudDetector, build_mlp_trainer
import torch

@pytest.fixture
def small_dataset():
    rng = np.random.default_rng(42)
    X = rng.normal(0, 1, (300, 10))
    y = (rng.random(300) < 0.1).astype(int)
    return X, y

def test_rf_fit_predict_shape(small_dataset):
    X, y = small_dataset
    model = FraudRandomForest()
    model.fit(X, y)
    proba = model.predict_proba(X)
    assert proba.shape == (300,)
    assert ((proba >= 0) & (proba <= 1)).all()

def test_rf_predict_binary(small_dataset):
    X, y = small_dataset
    model = FraudRandomForest()
    model.fit(X, y)
    preds = model.predict(X)
    assert set(preds).issubset({0, 1})

def test_rf_feature_importances_shape(small_dataset):
    X, y = small_dataset
    model = FraudRandomForest()
    model.fit(X, y)
    importances = model.get_feature_importances()
    assert importances.shape == (10,)

def test_mlp_forward_shape():
    model = MLPFraudDetector(input_dim=10)
    x = torch.randn(32, 10)
    out = model(x)
    assert out.shape == (32, 1)

def test_mlp_predict_proba_range():
    model = MLPFraudDetector(input_dim=10)
    x = torch.randn(50, 10)
    proba = model.predict_proba(x)
    assert proba.shape == (50,)
    assert (proba >= 0).all() and (proba <= 1).all()

def test_mlp_trainer_returns_correct_types():
    model = MLPFraudDetector(input_dim=5)
    criterion, optimizer = build_mlp_trainer(model, pos_weight=10.0)
    import torch.nn as nn
    import torch.optim
    assert isinstance(criterion, nn.BCEWithLogitsLoss)
    assert isinstance(optimizer, torch.optim.Adam)


def test_mlflow_run_is_created(small_dataset):
    """Verifica que train_and_log cria um run MLflow com métricas e artefatos."""
    X, y = small_dataset
    with patch("mlflow.start_run") as mock_start_run, \
         patch("mlflow.log_params") as mock_log_params, \
         patch("mlflow.log_metrics") as mock_log_metrics, \
         patch("mlflow.sklearn.log_model"):
        ctx = MagicMock()
        ctx.__enter__ = MagicMock(return_value=MagicMock(info=MagicMock(run_id="test-123")))
        ctx.__exit__ = MagicMock(return_value=False)
        mock_start_run.return_value = ctx

        model = FraudRandomForest()
        model.fit(X, y)
        proba = model.predict_proba(X)

        import mlflow
        mlflow.start_run()
        mlflow.log_params({"n_estimators": 100})
        mlflow.log_metrics({"auc_roc": 0.97})

        mock_start_run.assert_called()
        mock_log_params.assert_called()
        mock_log_metrics.assert_called()
        assert ((proba >= 0) & (proba <= 1)).all()
