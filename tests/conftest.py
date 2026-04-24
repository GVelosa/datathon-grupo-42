"""Fixtures compartilhadas para a suíte de testes do Datathon Grupo 42."""

from __future__ import annotations

import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock


@pytest.fixture
def sample_transactions_df() -> pd.DataFrame:
    """DataFrame sintético de transações com 200 registros e distribuição realista."""
    n = 200
    rng = np.random.default_rng(42)
    df = pd.DataFrame({
        "Time": rng.uniform(0, 172800, n),
        "Amount": rng.exponential(88, n),
        "Class": np.where(rng.random(n) < 0.02, 1, 0),
    })
    for i in range(1, 29):
        df[f"V{i}"] = rng.normal(0, 1, n)
    return df


@pytest.fixture
def mock_mlflow_client() -> MagicMock:
    """Mock do cliente MLflow para testes que verificam tracking sem servidor real."""
    client = MagicMock()
    run = MagicMock()
    run.info.run_id = "test-run-id-12345"
    run.info.experiment_id = "0"
    run.data.metrics = {"auc_roc": 0.97}
    run.data.params = {"n_estimators": "100"}
    client.create_run.return_value = run
    client.get_run.return_value = run
    return client


@pytest.fixture
def mock_anthropic_client() -> MagicMock:
    """Mock do cliente Anthropic que retorna end_turn com resposta de texto simples."""
    client = MagicMock()
    response = MagicMock()
    response.stop_reason = "end_turn"
    response.usage.input_tokens = 100
    response.usage.output_tokens = 50
    text_block = MagicMock()
    text_block.text = "Resposta de teste do agente."
    text_block.type = "text"
    response.content = [text_block]
    client.messages.create.return_value = response
    return client


@pytest.fixture
def sample_golden_pairs() -> list[dict]:
    """Pares query/answer mínimos para testes de avaliação."""
    return [
        {
            "id": "GS-001",
            "query": "Qual o AUC atual?",
            "expected_answer": "AUC-ROC de 0.9743.",
            "category": "metricas",
        },
        {
            "id": "GS-002",
            "query": "Ha drift nos dados?",
            "expected_answer": "PSI medio de 0.04, sem drift.",
            "category": "monitoramento",
        },
    ]
