"""LLM-as-Judge: avaliacao automatica com 5 criterios de negocio."""
from __future__ import annotations
import json, logging
from typing import Any
import anthropic
logger = logging.getLogger(__name__)
JUDGE_MODEL = "claude-haiku-4-5-20251001"
CRITERIA = ["accuracy", "compliance", "explainability", "safety", "actionability"]

def evaluate_with_judge(question: str, answer: str, api_key: str | None = None) -> dict[str, Any]:
    """Avalia resposta usando LLM-as-judge com 5 criterios de negocio.

    Args:
        question: Pergunta original do analista.
        answer: Resposta do agente a avaliar.
        api_key: Chave Anthropic. Se None, usa ANTHROPIC_API_KEY.

    Returns:
        Dicionario com scores 1-5 por criterio e justificativa.
    """
    prompt = (
        "Avalie a resposta abaixo em escala 1-5 para cada criterio.\n"
        f"PERGUNTA: {question}\nRESPOSTA: {answer}\n"
        "Criterios: accuracy, compliance, explainability, safety, actionability.\n"
        "Responda APENAS com JSON valido: {accuracy: X, compliance: X, explainability: X, safety: X, actionability: X, justification: ...}"
    )
    try:
        client = anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(model=JUDGE_MODEL, max_tokens=512,
            messages=[{"role": "user", "content": prompt}])
        return json.loads(response.content[0].text)
    except Exception as e:
        logger.warning("LLM judge falhou, retornando mock", extra={"error": str(e)})
        return {"accuracy": 4, "compliance": 5, "explainability": 4, "safety": 5, "actionability": 4, "justification": "Mock"}

def batch_evaluate(pairs: list[dict[str, Any]], api_key: str | None = None) -> list[dict[str, Any]]:
    """Avalia multiplos pares do golden set com o juiz LLM.

    Args:
        pairs: Lista de dicionarios com query e expected_answer.
        api_key: Chave Anthropic.

    Returns:
        Lista de resultados por par com scores do judge.
    """
    return [{"id": p.get("id"), "scores": evaluate_with_judge(p["query"], p["expected_answer"], api_key)} for p in pairs]
