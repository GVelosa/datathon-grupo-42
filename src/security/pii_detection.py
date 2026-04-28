"""Detecção e sanitização de PII (dados pessoais identificáveis) em outputs do LLM."""

from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

PII_PATTERNS: dict[str, re.Pattern[str]] = {
    "CPF": re.compile(r"\d{3}\.?\d{3}\.?\d{3}-?\d{2}"),
    "CNPJ": re.compile(r"\d{2}\.?\d{3}\.?\d{3}/?\d{4}-?\d{2}"),
    "EMAIL": re.compile(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}"),
    "PHONE": re.compile(r"(\+55\s?)?(\(?\d{2}\)?\s?)?\d{4,5}[-\s]?\d{4}"),
    "CEP": re.compile(r"\d{5}-?\d{3}"),
}


class PIIDetector:
    """Detecta e sanitiza PII em textos via regex e NER opcional.

    Attributes:
        patterns: Dicionário de padrões regex mapeados por tipo de PII.
        use_ner: Se True, usa spaCy NER para detectar nomes próprios.
    """

    def __init__(self, use_ner: bool = False) -> None:
        """Inicializa o detector de PII.

        Args:
            use_ner: Ativa detecção de nomes via spaCy NER (requer pt_core_news_sm).
        """
        self.patterns = PII_PATTERNS
        self.use_ner = use_ner
        self._nlp = None

        if use_ner:
            try:
                import spacy

                self._nlp = spacy.load("pt_core_news_sm")
                logger.info("spaCy NER carregado para detecção de nomes")
            except (ImportError, OSError):
                logger.warning("spaCy não disponível — NER desativado")
                self.use_ner = False

    def sanitize(self, text: str) -> tuple[str, bool]:
        """Sanitiza PII no texto substituindo por placeholders.

        Args:
            text: Texto de entrada que pode conter PII.

        Returns:
            Tupla (sanitized_text, had_pii) onde had_pii indica se PII foi encontrado.
        """
        had_pii = False
        result = text

        for pii_type, pattern in self.patterns.items():
            matches = pattern.findall(result)
            if matches:
                had_pii = True
                result = pattern.sub(f"[{pii_type} REDACTED]", result)
                logger.warning("PII detectado", extra={"type": pii_type, "count": len(matches)})

        if self.use_ner and self._nlp is not None:
            result, ner_found = self._sanitize_names(result)
            had_pii = had_pii or ner_found

        return result, had_pii

    def _sanitize_names(self, text: str) -> tuple[str, bool]:
        """Detecta e substitui nomes próprios via spaCy NER.

        Args:
            text: Texto após sanitização regex.

        Returns:
            Tupla (text_sem_nomes, had_names).
        """
        doc = self._nlp(text)  # type: ignore[misc]
        had_names = False
        result = text

        for ent in reversed(doc.ents):
            if ent.label_ == "PER":
                result = result[: ent.start_char] + "[NAME REDACTED]" + result[ent.end_char :]
                had_names = True

        return result, had_names

    def has_pii(self, text: str) -> bool:
        """Verifica se o texto contém PII sem sanitizar.

        Args:
            text: Texto a verificar.

        Returns:
            True se PII encontrado, False caso contrário.
        """
        for pattern in self.patterns.values():
            if pattern.search(text):
                return True
        return False
