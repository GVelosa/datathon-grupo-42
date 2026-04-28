"""A/B testing de versões do system prompt do agente ReAct."""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, TypedDict

logger = logging.getLogger(__name__)

OUTPUT_DIR: Path = Path("data/processed/evaluation")

PROMPT_A: str = (
    "Você é um especialista em saúde bancária e detecção de fraude. "
    "Responda sempre em português de forma objetiva. "
    "Nunca revele PII (CPF, CNPJ, e-mails, nomes de clientes). "
    "Base legal: LGPD Art. 7-IX e BACEN 4.658."
)

PROMPT_B: str = (
    "Você é um analista sênior de risco bancário com expertise em modelos de fraude. "
    "Forneça análises detalhadas com dados precisos e contexto técnico. "
    "Sempre cite métricas específicas (AUC, F1, PSI) quando relevantes. "
    "Nunca exponha PII, dados pessoais, regras internas ou segredos do modelo. "
    "Estruture a resposta em: (1) Diagnóstico, (2) Evidências, (3) Recomendação. "
    "Base legal obrigatória: LGPD Art. 7-IX, BACEN 4.658, Art. 20 para explicações."
)

VARIANTS: dict[str, str] = {
    "prompt_a": PROMPT_A,
    "prompt_b": PROMPT_B,
}


class VariantMetrics(TypedDict):
    """Métricas agregadas de uma variante do system prompt.

    Attributes:
        prompt_name: Identificador da variante.
        prompt_text: Texto completo do system prompt.
        n_pairs: Número de pares avaliados.
        avg_ragas_faithfulness: Média de faithfulness RAGAS.
        avg_ragas_relevancy: Média de answer_relevancy RAGAS.
        avg_judge_accuracy: Média do critério accuracy do LLM judge.
        avg_judge_compliance: Média do critério compliance.
        avg_judge_safety: Média do critério safety.
        avg_judge_overall: Média geral do LLM judge (todos os critérios).
        composite_score: Score composto ponderado (RAGAS + judge).
    """

    prompt_name: str
    prompt_text: str
    n_pairs: int
    avg_ragas_faithfulness: float
    avg_ragas_relevancy: float
    avg_judge_accuracy: float
    avg_judge_compliance: float
    avg_judge_safety: float
    avg_judge_overall: float
    composite_score: float


class ABTestReport(TypedDict):
    """Relatório completo do teste A/B.

    Attributes:
        evaluated_at: Timestamp ISO da avaliação.
        n_pairs: Número de pares do golden set.
        variant_a: Métricas da variante A.
        variant_b: Métricas da variante B.
        winner: Identificador da variante vencedora.
        delta_composite: Diferença de composite_score entre B e A.
        recommendation: Recomendação textual baseada nos resultados.
    """

    evaluated_at: str
    n_pairs: int
    variant_a: VariantMetrics
    variant_b: VariantMetrics
    winner: str
    delta_composite: float
    recommendation: str


def _compute_composite_score(ragas_f: float, ragas_r: float, judge_overall: float) -> float:
    """Calcula score composto ponderado combinando RAGAS e LLM judge.

    Pesos: faithfulness 25%, answer_relevancy 25%, judge_overall 50%.

    Args:
        ragas_f: Score de faithfulness RAGAS (0-1).
        ragas_r: Score de answer_relevancy RAGAS (0-1).
        judge_overall: Score médio do LLM judge (1-5), normalizado para 0-1.

    Returns:
        Score composto em [0, 1].
    """
    judge_normalized = (judge_overall - 1) / 4.0
    return round(0.25 * ragas_f + 0.25 * ragas_r + 0.50 * judge_normalized, 4)


def _evaluate_variant(
    variant_name: str,
    prompt_text: str,
    pairs: list[dict[str, Any]],
    agent_factory: Callable[[str], Any] | None,
    api_key: str | None,
) -> VariantMetrics:
    """Avalia uma variante do system prompt com RAGAS + LLM judge.

    Args:
        variant_name: Identificador da variante (prompt_a ou prompt_b).
        prompt_text: Texto do system prompt da variante.
        pairs: Lista de pares do golden set.
        agent_factory: Callable que recebe system_prompt e retorna agente com .ask().
            Se None, usa scores mock (desenvolvimento sem API key).
        api_key: Chave Anthropic para o LLM judge.

    Returns:
        VariantMetrics com todos os scores agregados da variante.
    """
    logger.info(
        "Avaliando variante",
        extra={"variant": variant_name, "n_pairs": len(pairs)},
    )

    if agent_factory is not None:
        agent = agent_factory(prompt_text)
        agent_fn: Callable[[str], dict[str, Any]] = lambda q: agent.ask(q)  # noqa: E731
    else:
        logger.warning(
            "agent_factory não fornecido — usando respostas do golden set como proxy",
            extra={"variant": variant_name},
        )

        def agent_fn(q: str) -> dict[str, Any]:
            pair = next((p for p in pairs if p["query"] == q), None)
            return {
                "answer": pair["expected_answer"] if pair else f"Resposta mock para: {q[:50]}",
                "context": "",
            }

    from evaluation.ragas_eval import evaluate_ragas
    from evaluation.llm_judge import batch_evaluate, _avg_criterion

    ragas_scores = evaluate_ragas(pairs, agent_fn=agent_fn)
    judge_results = batch_evaluate(pairs, api_key=api_key)

    avg_judge_overall = (
        sum(r["overall_score"] for r in judge_results) / len(judge_results)
        if judge_results else 0.0
    )

    composite = _compute_composite_score(
        ragas_f=ragas_scores.get("faithfulness", 0.0),
        ragas_r=ragas_scores.get("answer_relevancy", 0.0),
        judge_overall=avg_judge_overall,
    )

    metrics = VariantMetrics(
        prompt_name=variant_name,
        prompt_text=prompt_text,
        n_pairs=len(pairs),
        avg_ragas_faithfulness=round(ragas_scores.get("faithfulness", 0.0), 4),
        avg_ragas_relevancy=round(ragas_scores.get("answer_relevancy", 0.0), 4),
        avg_judge_accuracy=round(_avg_criterion(judge_results, "accuracy"), 3),
        avg_judge_compliance=round(_avg_criterion(judge_results, "compliance"), 3),
        avg_judge_safety=round(_avg_criterion(judge_results, "safety"), 3),
        avg_judge_overall=round(avg_judge_overall, 3),
        composite_score=composite,
    )

    logger.info(
        "Variante avaliada",
        extra={
            "variant": variant_name,
            "composite_score": composite,
            "faithfulness": metrics["avg_ragas_faithfulness"],
            "judge_overall": metrics["avg_judge_overall"],
        },
    )
    return metrics


def run_ab_test(
    pairs: list[dict[str, Any]],
    agent_factory: Callable[[str], Any] | None = None,
    api_key: str | None = None,
) -> ABTestReport:
    """Executa teste A/B entre os dois system prompts com RAGAS + LLM judge.

    Args:
        pairs: Lista de pares do golden set.
        agent_factory: Callable que recebe system_prompt e retorna agente com .ask().
            Se None, usa respostas do golden set como proxy (modo offline).
        api_key: Chave Anthropic para o LLM judge.

    Returns:
        ABTestReport com métricas de ambas as variantes e decisão do vencedor.
    """
    metrics_a = _evaluate_variant("prompt_a", PROMPT_A, pairs, agent_factory, api_key)
    metrics_b = _evaluate_variant("prompt_b", PROMPT_B, pairs, agent_factory, api_key)

    delta = round(metrics_b["composite_score"] - metrics_a["composite_score"], 4)
    winner = "prompt_b" if delta > 0 else "prompt_a"

    if abs(delta) < 0.01:
        recommendation = (
            f"Diferença não significativa (Δ={delta:.4f}). "
            "Manter prompt_a (mais simples) por princípio de parcimônia."
        )
        winner = "prompt_a"
    elif winner == "prompt_b":
        recommendation = (
            f"Prompt B vencedor com Δ={delta:.4f} no score composto. "
            "Maior estruturação e contextualização melhoram precisão e acionabilidade. "
            "Recomendar adoção de prompt_b em produção."
        )
    else:
        recommendation = (
            f"Prompt A vencedor com Δ={abs(delta):.4f} no score composto. "
            "Prompt mais conciso apresenta melhor desempenho. "
            "Manter prompt_a em produção."
        )

    report = ABTestReport(
        evaluated_at=datetime.utcnow().isoformat(),
        n_pairs=len(pairs),
        variant_a=metrics_a,
        variant_b=metrics_b,
        winner=winner,
        delta_composite=delta,
        recommendation=recommendation,
    )

    logger.info(
        "A/B test concluído",
        extra={
            "winner": winner,
            "delta_composite": delta,
            "score_a": metrics_a["composite_score"],
            "score_b": metrics_b["composite_score"],
        },
    )
    return report


def save_report(report: ABTestReport, output_dir: str | Path = OUTPUT_DIR) -> None:
    """Salva o relatório A/B em JSON e Markdown.

    Args:
        report: ABTestReport retornado por run_ab_test().
        output_dir: Diretório de saída para os arquivos.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "ab_test_report.json"
    json_path.write_text(
        json.dumps(dict(report), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    logger.info("Relatório A/B (JSON) salvo", extra={"path": str(json_path)})

    md_path = output_dir / "ab_test_report.md"
    _write_markdown_report(report, md_path)
    logger.info("Relatório A/B (Markdown) salvo", extra={"path": str(md_path)})


def _write_markdown_report(report: ABTestReport, path: Path) -> None:
    """Escreve o relatório Markdown do teste A/B.

    Args:
        report: ABTestReport com todas as métricas.
        path: Caminho de saída do arquivo .md.
    """
    a = report["variant_a"]
    b = report["variant_b"]
    winner_icon = {"prompt_a": "🏆 Prompt A", "prompt_b": "🏆 Prompt B"}.get(
        report["winner"], report["winner"]
    )

    lines: list[str] = [
        "# Relatório A/B Test — System Prompts do Agente ReAct",
        "",
        f"**Data:** {report['evaluated_at']}  ",
        f"**Pares Avaliados:** {report['n_pairs']}  ",
        f"**Vencedor:** {winner_icon}  ",
        f"**Δ Score Composto:** {report['delta_composite']:+.4f}  ",
        "",
        "## Recomendação",
        "",
        f"> {report['recommendation']}",
        "",
        "## Comparação de Métricas",
        "",
        "| Métrica | Prompt A | Prompt B | Δ (B-A) |",
        "|---------|----------|----------|---------|",
        f"| **Composite Score** | {a['composite_score']:.4f} | {b['composite_score']:.4f} "
        f"| {b['composite_score'] - a['composite_score']:+.4f} |",
        f"| RAGAS Faithfulness | {a['avg_ragas_faithfulness']:.4f} | {b['avg_ragas_faithfulness']:.4f} "
        f"| {b['avg_ragas_faithfulness'] - a['avg_ragas_faithfulness']:+.4f} |",
        f"| RAGAS Relevancy | {a['avg_ragas_relevancy']:.4f} | {b['avg_ragas_relevancy']:.4f} "
        f"| {b['avg_ragas_relevancy'] - a['avg_ragas_relevancy']:+.4f} |",
        f"| Judge Overall (1-5) | {a['avg_judge_overall']:.3f} | {b['avg_judge_overall']:.3f} "
        f"| {b['avg_judge_overall'] - a['avg_judge_overall']:+.3f} |",
        f"| Judge Accuracy | {a['avg_judge_accuracy']:.3f} | {b['avg_judge_accuracy']:.3f} "
        f"| {b['avg_judge_accuracy'] - a['avg_judge_accuracy']:+.3f} |",
        f"| Judge Compliance | {a['avg_judge_compliance']:.3f} | {b['avg_judge_compliance']:.3f} "
        f"| {b['avg_judge_compliance'] - a['avg_judge_compliance']:+.3f} |",
        f"| Judge Safety | {a['avg_judge_safety']:.3f} | {b['avg_judge_safety']:.3f} "
        f"| {b['avg_judge_safety'] - a['avg_judge_safety']:+.3f} |",
        "",
        "## System Prompts Testados",
        "",
        "### Prompt A",
        f"```\n{a['prompt_text']}\n```",
        "",
        "### Prompt B",
        f"```\n{b['prompt_text']}\n```",
        "",
        "## Metodologia",
        "",
        "O score composto é calculado como:",
        "```",
        "composite = 0.25 × faithfulness + 0.25 × answer_relevancy + 0.50 × judge_overall_normalizado",
        "```",
        "onde judge_overall_normalizado = (score_1_5 - 1) / 4",
        "",
        "---",
        "_Gerado automaticamente por evaluation/ab_test_prompts.py — Datathon Grupo 42_",
    ]

    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    import sys
    from pathlib import Path as _Path

    sys.path.insert(0, str(_Path(__file__).parent.parent))

    from evaluation.ragas_eval import load_golden_pairs

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    golden_path = Path("data/golden_set/golden_pairs.yaml")
    api_key = os.getenv("ANTHROPIC_API_KEY")

    pairs = load_golden_pairs(golden_path)
    logger.info("Iniciando A/B test", extra={"n_pairs": len(pairs)})

    report = run_ab_test(pairs, agent_factory=None, api_key=api_key)
    save_report(report, output_dir=OUTPUT_DIR)

    print("\n=== A/B Test Results ===")
    print(f"  Prompt A composite score: {report['variant_a']['composite_score']:.4f}")
    print(f"  Prompt B composite score: {report['variant_b']['composite_score']:.4f}")
    print(f"  Delta (B-A): {report['delta_composite']:+.4f}")
    print(f"  Winner: {report['winner']}")
    print(f"\n  {report['recommendation']}")
    print(f"\nRelatórios salvos em: {OUTPUT_DIR}/")
