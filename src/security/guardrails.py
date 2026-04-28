"""Guardrails de segurança: detecção de prompt injection e sanitização de output."""

from __future__ import annotations

import logging
import re
from typing import TypedDict

from src.security.pii_detection import PIIDetector

logger = logging.getLogger(__name__)

INJECTION_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"ignore\s+(\w+\s+)*instructions", re.IGNORECASE),
    re.compile(r"(jailbreak|DAN|do\s+anything\s+now)", re.IGNORECASE),
    re.compile(r"(reveal|show|print|output)\s+(system\s+prompt|training\s+data|internal)", re.IGNORECASE),
    re.compile(r"act\s+as\s+(if\s+you|an\s+AI\s+without|a\s+different)", re.IGNORECASE),
    re.compile(r"(bypass|override|disable)\s+(your|all)\s+(restrictions|filters|safety)", re.IGNORECASE),
    re.compile(r"forget\s+(everything|all)\s+(you|I)\s+(told|said)", re.IGNORECASE),
    re.compile(r"(mostre|revele|liste|exporte)\s+(todos\s+os|os)\s+(registros|dados|clientes)", re.IGNORECASE),
    re.compile(r"(dump|extract)\s+(database|tabela|dados\s+internos)", re.IGNORECASE),
]

MAX_INPUT_LENGTH: int = 1000


class GuardrailResult(TypedDict):
    """Resultado da verificação de guardrail.

    Attributes:
        allowed: Se a requisição foi permitida.
        reason: Motivo da decisão ('ok' ou categoria de bloqueio).
        category: Categoria OWASP LLM se bloqueado, None se permitido.
    """

    allowed: bool
    reason: str
    category: str | None


class InputGuardrail:
    """Verifica inputs antes de enviá-los ao LLM.

    Detecta tentativas de prompt injection via regex patterns.
    Bloqueia inputs acima do comprimento máximo permitido.
    """

    def __init__(self, max_length: int = MAX_INPUT_LENGTH) -> None:
        """Inicializa o guardrail de input.

        Args:
            max_length: Comprimento máximo permitido para o input em caracteres.
        """
        self.patterns = INJECTION_PATTERNS
        self.max_length = max_length

    def check(self, text: str) -> GuardrailResult:
        """Verifica se o input é seguro para processar.

        Args:
            text: Input do usuário a verificar.

        Returns:
            GuardrailResult com allowed, reason e category OWASP.
        """
        if len(text) > self.max_length:
            logger.warning("Input excede comprimento máximo", extra={"length": len(text)})
            return GuardrailResult(allowed=False, reason="input_too_long", category="LLM04")

        for pattern in self.patterns:
            if pattern.search(text):
                logger.warning(
                    "Prompt injection detectada",
                    extra={"pattern": pattern.pattern[:50], "input_preview": text[:100]},
                )
                return GuardrailResult(allowed=False, reason="prompt_injection", category="LLM01")

        return GuardrailResult(allowed=True, reason="ok", category=None)


class OutputGuardrail:
    """Sanitiza outputs do LLM antes de retornar ao usuário.

    Aplica PIIDetector para remover dados pessoais identificáveis.
    """

    def __init__(self, use_ner: bool = False) -> None:
        """Inicializa o guardrail de output.

        Args:
            use_ner: Ativa detecção de nomes próprios via spaCy NER.
        """
        self.pii_detector = PIIDetector(use_ner=use_ner)

    def apply(self, text: str) -> tuple[str, bool]:
        """Sanitiza o output removendo PII.

        Args:
            text: Output bruto do LLM.

        Returns:
            Tupla (sanitized_text, had_pii) indicando se PII foi encontrado e removido.
        """
        sanitized, had_pii = self.pii_detector.sanitize(text)
        if had_pii:
            logger.warning("PII removido do output do LLM")
        return sanitized, had_pii
