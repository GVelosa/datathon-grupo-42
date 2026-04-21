"""Modelos baseline: RandomForestClassifier (sklearn) e MLPFraudDetector (PyTorch)."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class FraudRandomForest:
    """Wrapper para RandomForestClassifier com configurações padrão para fraude.

    Attributes:
        model: Instância do RandomForestClassifier configurado.
        scaler: StandardScaler fitado durante o treino.
        threshold: Threshold de decisão calibrado via Precision-Recall.
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        """Inicializa o modelo com configurações padrão ou fornecidas.

        Args:
            config: Dicionário com hiperparâmetros. Usa defaults se None.
        """
        cfg = config or {}
        self.model = RandomForestClassifier(
            n_estimators=cfg.get("n_estimators", 200),
            max_depth=cfg.get("max_depth", 15),
            class_weight=cfg.get("class_weight", "balanced"),
            min_samples_leaf=cfg.get("min_samples_leaf", 10),
            random_state=cfg.get("random_state", 42),
            n_jobs=-1,
        )
        self.scaler = StandardScaler()
        self.threshold: float = cfg.get("decision_threshold", 0.35)

    def fit(self, X: np.ndarray, y: np.ndarray) -> FraudRandomForest:
        """Treina o modelo.

        Args:
            X: Features de treino com shape (n_samples, n_features).
            y: Labels binários com shape (n_samples,).

        Returns:
            Self para encadeamento.
        """
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        logger.info("RandomForest treinado", extra={"n_trees": self.model.n_estimators, "samples": len(X)})
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Retorna probabilidades de fraude.

        Args:
            X: Features com shape (n_samples, n_features).

        Returns:
            Array de probabilidades de fraude com shape (n_samples,).
        """
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)[:, 1]

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Retorna predições binárias usando o threshold calibrado.

        Args:
            X: Features com shape (n_samples, n_features).

        Returns:
            Array binário com shape (n_samples,): 1=fraude, 0=legítimo.
        """
        return (self.predict_proba(X) >= self.threshold).astype(int)

    def get_feature_importances(self) -> np.ndarray:
        """Retorna importância das features pelo modelo.

        Returns:
            Array de importâncias com shape (n_features,).
        """
        return self.model.feature_importances_


class MLPFraudDetector(nn.Module):
    """MLP para detecção de fraude implementado em PyTorch.

    Attributes:
        network: Rede neural sequencial com camadas ocultas e dropout.
    """

    def __init__(self, input_dim: int = 33, hidden_layers: list[int] | None = None, dropout: float = 0.3) -> None:
        """Inicializa a arquitetura MLP.

        Args:
            input_dim: Dimensão do vetor de entrada (default 31 features).
            hidden_layers: Lista com tamanhos das camadas ocultas. Default: [128, 64, 32].
            dropout: Taxa de dropout para regularização.
        """
        super().__init__()
        if hidden_layers is None:
            hidden_layers = [128, 64, 32]

        layers: list[nn.Module] = []
        prev_dim = input_dim
        for h_dim in hidden_layers:
            layers.extend([nn.Linear(prev_dim, h_dim), nn.ReLU(), nn.Dropout(dropout)])
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, 1))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass do MLP.

        Args:
            x: Tensor de entrada com shape (batch_size, input_dim).

        Returns:
            Logits com shape (batch_size, 1).
        """
        return self.network(x)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Retorna probabilidades via sigmoid.

        Args:
            x: Tensor de entrada com shape (batch_size, input_dim).

        Returns:
            Tensor de probabilidades com shape (batch_size,) em [0, 1].
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            return torch.sigmoid(logits).squeeze(-1)


def build_mlp_trainer(
    model: MLPFraudDetector,
    pos_weight: float = 577.0,
    lr: float = 1e-3,
) -> tuple[nn.BCEWithLogitsLoss, torch.optim.Optimizer]:
    """Cria loss e optimizer configurados para dados desbalanceados.

    Args:
        model: Instância do MLPFraudDetector.
        pos_weight: Peso para a classe positiva (fraude). Default: 577 (ratio do dataset).
        lr: Learning rate do Adam. Default: 0.001.

    Returns:
        Tupla (criterion, optimizer) prontos para o loop de treino.
    """
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    return criterion, optimizer
