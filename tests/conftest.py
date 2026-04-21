import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock

@pytest.fixture
def sample_transactions_df():
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
def mock_anthropic_client():
    client = MagicMock()
    response = MagicMock()
    response.stop_reason = "end_turn"
    text_block = MagicMock()
    text_block.text = "Resposta de teste do agente."
    response.content = [text_block]
    client.messages.create.return_value = response
    return client

@pytest.fixture
def sample_golden_pairs():
    return [
        {"id": "GS-001", "query": "Qual o AUC atual?", "expected_answer": "AUC-ROC de 0.9743.", "category": "metricas"},
        {"id": "GS-002", "query": "Ha drift nos dados?", "expected_answer": "PSI medio de 0.04, sem drift.", "category": "monitoramento"},
    ]
