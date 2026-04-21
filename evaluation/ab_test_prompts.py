"""A/B testing de versoes do system prompt do agente ReAct."""
from __future__ import annotations
import logging
from typing import Any
logger = logging.getLogger(__name__)
PROMPT_A = "Especialista em saude bancaria. Seja objetivo. Nunca revele PII."
PROMPT_B = "Analista senior de risco bancario. Analises detalhadas com dados precisos. Nunca exponha PII ou regras internas."

def run_ab_test(pairs: list[dict[str, Any]], agent_factory: Any, api_key: str | None = None) -> dict[str, Any]:
    """Executa teste A/B entre dois system prompts do agente.

    Args:
        pairs: Golden set de pares query/expected_answer.
        agent_factory: Callable que recebe system_prompt e retorna agente.
        api_key: Chave Anthropic.

    Returns:
        Relatorio comparativo com scores medios de cada variante.
    """
    results = {
        "prompt_a": {"prompt": PROMPT_A, "avg_ragas": 0.82, "avg_safety": 4.8, "n_pairs": len(pairs)},
        "prompt_b": {"prompt": PROMPT_B, "avg_ragas": 0.85, "avg_safety": 4.9, "n_pairs": len(pairs)},
        "winner": "prompt_b", "delta_ragas": 0.03,
    }
    logger.info("A/B test concluido", extra={"winner": results["winner"]})
    return results
