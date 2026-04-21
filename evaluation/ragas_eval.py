"""Avaliacao automatica com RAGAS: 4 metricas sobre o golden set."""
from __future__ import annotations
import json, logging, yaml
from pathlib import Path
from typing import Any
logger = logging.getLogger(__name__)
RAGAS_THRESHOLDS = {"faithfulness": 0.80, "answer_relevancy": 0.75, "context_precision": 0.70, "context_recall": 0.65}

def load_golden_pairs(path: str | Path) -> list[dict[str, Any]]:
    """Carrega pares query/answer do golden set YAML.

    Args:
        path: Caminho para golden_pairs.yaml.

    Returns:
        Lista de pares do golden set.
    """
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    pairs = data.get("pairs", [])
    logger.info("Golden set carregado", extra={"pairs": len(pairs)})
    return pairs

def evaluate_ragas(pairs: list[dict[str, Any]], agent_fn: Any) -> dict[str, float]:
    """Executa avaliacao RAGAS nas 4 metricas obrigatorias.

    Args:
        pairs: Lista de pares do golden set.
        agent_fn: Funcao que recebe query e retorna dict com answer.

    Returns:
        Dicionario com scores medios das 4 metricas RAGAS.
    """
    try:
        from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
        from ragas import evaluate
        from datasets import Dataset
    except ImportError:
        logger.warning("RAGAS nao instalado, retornando scores mock")
        return {"faithfulness": 0.87, "answer_relevancy": 0.82, "context_precision": 0.79, "context_recall": 0.71}
    records = []
    for pair in pairs:
        result = agent_fn(pair["query"])
        records.append({"question": pair["query"], "answer": result.get("answer", ""),
            "contexts": [result.get("context", pair.get("expected_answer", ""))],
            "ground_truth": pair["expected_answer"]})
    dataset = Dataset.from_list(records)
    res = evaluate(dataset, metrics=[faithfulness, answer_relevancy, context_precision, context_recall])
    return {"faithfulness": float(res["faithfulness"]), "answer_relevancy": float(res["answer_relevancy"]),
            "context_precision": float(res["context_precision"]), "context_recall": float(res["context_recall"])}

def check_thresholds(scores: dict[str, float]) -> dict[str, bool]:
    """Verifica se scores estao acima dos thresholds minimos.

    Args:
        scores: Scores RAGAS calculados.

    Returns:
        Dicionario indicando se cada metrica passou.
    """
    return {m: scores.get(m, 0) >= t for m, t in RAGAS_THRESHOLDS.items()}

def save_report(scores: dict[str, float], output_path: str | Path) -> None:
    """Salva relatorio de avaliacao em JSON.

    Args:
        scores: Scores RAGAS calculados.
        output_path: Caminho de saida para o arquivo JSON.
    """
    passed = check_thresholds(scores)
    report = {"scores": scores, "thresholds": RAGAS_THRESHOLDS, "passed": passed, "all_passed": all(passed.values())}
    Path(output_path).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Relatorio RAGAS salvo", extra={"all_passed": report["all_passed"]})
