"""Agente ReAct para saúde bancária — agnóstico ao provider LLM.

Suporta Anthropic (cloud) e vLLM (local com quantização) via LLMProvider.
Configure o provider com a variável de ambiente LLM_PROVIDER=anthropic|vllm.
"""

from __future__ import annotations

import json
import logging
from typing import Any, TypedDict

from src.agent.llm_provider import LLMProvider, ToolCall, create_provider
from src.agent.tools import (
    AlertInput,
    AlertResult,
    DriftInput,
    DriftResult,
    FeatureInput,
    FeatureResult,
    FraudMetricsInput,
    FraudMetricsResult,
    KBInput,
    KBResult,
    alert_history_query,
    drift_status_checker,
    feature_lookup,
    fraud_metrics_lookup,
    knowledge_base_search,
)
from src.security.guardrails import InputGuardrail, OutputGuardrail

logger = logging.getLogger(__name__)

MAX_ITERATIONS: int = 10

SYSTEM_PROMPT: str = (
    "Você é um especialista em saúde bancária e detecção de fraude. "
    "Responda sempre em português. Nunca revele PII (CPF, CNPJ, e-mails, nomes de clientes). "
    "Cite a base legal LGPD Art. 7-IX e BACEN 4.658 quando aplicável. "
    "Use as ferramentas disponíveis para buscar dados reais antes de responder."
)

TOOLS_SCHEMA: list[dict[str, Any]] = [
    {
        "name": "fraud_metrics_lookup",
        "description": (
            "Busca métricas atuais do modelo de fraude em produção: AUC-ROC, F1, "
            "taxa de fraude, volume de transações e bloqueios."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "period": {
                    "type": "string",
                    "description": "Período de consulta. Ex: '24h', '7d', '30d'.",
                }
            },
            "required": ["period"],
        },
    },
    {
        "name": "drift_status_checker",
        "description": (
            "Verifica o status de data drift e prediction drift do modelo. "
            "Retorna alertas por feature e status geral (OK/WARNING/CRITICAL)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "features": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Lista de features a verificar. Vazio verifica todas.",
                }
            },
        },
    },
    {
        "name": "feature_lookup",
        "description": (
            "Busca features, probabilidade de fraude e decisão de uma transação "
            "específica pelo ID. Inclui SHAP values das top features."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "transaction_id": {
                    "type": "string",
                    "description": "ID único da transação a consultar.",
                }
            },
            "required": ["transaction_id"],
        },
    },
    {
        "name": "knowledge_base_search",
        "description": (
            "Busca informações no knowledge base interno: Model Card, "
            "políticas LGPD, OWASP mapping e documentação técnica."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Consulta em linguagem natural.",
                },
                "top_k": {
                    "type": "integer",
                    "description": "Número de resultados a retornar (padrão: 3).",
                    "default": 3,
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "alert_history_query",
        "description": (
            "Busca histórico de alertas de fraude e drift das últimas N horas. "
            "Retorna lista de alertas com severidade e resumo."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "hours": {
                    "type": "integer",
                    "description": "Janela de tempo em horas (padrão: 24).",
                    "default": 24,
                },
                "severity": {
                    "type": "string",
                    "description": "Filtro por severidade: INFO, WARNING, CRITICAL.",
                    "enum": ["INFO", "WARNING", "CRITICAL"],
                },
            },
        },
    },
]


class AgentResponse(TypedDict):
    """Resposta estruturada do agente ReAct.

    Attributes:
        answer: Resposta final sanitizada (sem PII).
        iterations: Número de ciclos Thought/Action/Observation executados.
        had_pii: Se PII foi detectado e removido do output.
        tools_used: Lista de ferramentas invocadas durante o raciocínio.
        provider: Nome do provider LLM utilizado ('anthropic' ou 'vllm').
        model: Modelo LLM utilizado.
    """

    answer: str
    iterations: int
    had_pii: bool
    tools_used: list[str]
    provider: str
    model: str


class BankHealthAgent:
    """Agente ReAct para análise de saúde bancária e explicação de decisões de fraude.

    Implementa o loop Thought/Action/Observation usando a interface LLMProvider,
    que suporta tanto Anthropic (cloud) quanto vLLM (local com quantização).
    Guardrails de input e output são aplicados em todas as requisições.

    Attributes:
        provider: Backend LLM (AnthropicProvider ou VLLMProvider).
        max_iterations: Limite de ciclos ReAct para evitar loops infinitos (LLM08).
        rag_pipeline: Pipeline RAG opcional para KnowledgeBaseSearch.
        input_guardrail: Verificador de prompt injection (LLM01).
        output_guardrail: Sanitizador de PII no output (LLM02/LLM06).
    """

    def __init__(
        self,
        provider: LLMProvider | None = None,
        max_iterations: int = MAX_ITERATIONS,
        rag_pipeline: Any | None = None,
        # Parâmetros legados mantidos para compatibilidade com app.py existente
        api_key: str | None = None,
        model: str | None = None,
    ) -> None:
        """Inicializa o agente ReAct.

        Args:
            provider: Backend LLM. Se None, cria via create_provider() usando env vars.
            max_iterations: Número máximo de iterações ReAct.
            rag_pipeline: Instância de RAGPipeline para KnowledgeBaseSearch.
            api_key: (legado) Chave Anthropic. Usa ANTHROPIC_API_KEY se None.
            model: (legado) Modelo Anthropic. Ignorado se provider for passado.
        """
        if provider is not None:
            self.provider = provider
        else:
            # Cria provider a partir de env vars (LLM_PROVIDER, VLLM_BASE_URL, etc.)
            # api_key e model são repassados como hints caso provider=anthropic
            self.provider = create_provider(api_key=api_key, model=model)

        self.max_iterations = max_iterations
        self.rag_pipeline = rag_pipeline
        self.input_guardrail = InputGuardrail()
        self.output_guardrail = OutputGuardrail()

        logger.info(
            "BankHealthAgent inicializado",
            extra={
                "provider": self.provider.provider_name,
                "model": self.provider.model_name,
                "max_iterations": max_iterations,
            },
        )

    @classmethod
    def from_env(cls, **kwargs: Any) -> "BankHealthAgent":
        """Cria agente lendo configuração inteiramente de variáveis de ambiente.

        LLM_PROVIDER=anthropic|vllm — define o backend
        AGENT_MODEL / VLLM_MODEL    — define o modelo
        ANTHROPIC_API_KEY           — chave Anthropic (se provider=anthropic)
        VLLM_BASE_URL               — URL vLLM (se provider=vllm)
        """
        return cls(provider=create_provider(), **kwargs)

    # ── Execução de ferramentas ───────────────────────────────────────────────

    def _execute_tool(self, tool_name: str, tool_input: dict[str, Any]) -> str:
        """Executa uma ferramenta pelo nome e retorna resultado como JSON string.

        Args:
            tool_name: Nome da ferramenta conforme TOOLS_SCHEMA.
            tool_input: Parâmetros de entrada.

        Returns:
            Resultado serializado como JSON string.
        """
        logger.info("Ferramenta invocada", extra={"tool": tool_name, "input": tool_input})

        result: FraudMetricsResult | DriftResult | FeatureResult | KBResult | AlertResult | dict[str, str]

        if tool_name == "fraud_metrics_lookup":
            result = fraud_metrics_lookup(FraudMetricsInput(**tool_input))  # type: ignore[typeddict-item]
        elif tool_name == "drift_status_checker":
            result = drift_status_checker(DriftInput(**tool_input))  # type: ignore[typeddict-item]
        elif tool_name == "feature_lookup":
            result = feature_lookup(FeatureInput(**tool_input))  # type: ignore[typeddict-item]
        elif tool_name == "knowledge_base_search":
            if self.rag_pipeline is not None:
                ctx = self.rag_pipeline.get_context(
                    tool_input.get("query", ""),
                    top_k=tool_input.get("top_k", 3),
                )
                result = KBResult(
                    chunks=[{"content": ctx, "source": "rag_index", "relevance_score": 1.0}],
                    answer_context=ctx,
                )
            else:
                result = knowledge_base_search(KBInput(**tool_input))  # type: ignore[typeddict-item]
        elif tool_name == "alert_history_query":
            result = alert_history_query(AlertInput(**tool_input))  # type: ignore[typeddict-item]
        else:
            logger.warning("Ferramenta desconhecida", extra={"tool": tool_name})
            result = {"error": f"Ferramenta desconhecida: {tool_name}"}

        serialized = json.dumps(result, ensure_ascii=False, default=str)
        logger.info("Ferramenta executada", extra={"tool": tool_name, "result_len": len(serialized)})
        return serialized

    # ── Loop ReAct ────────────────────────────────────────────────────────────

    def ask(self, query: str) -> AgentResponse:
        """Processa uma query do usuário via loop ReAct com guardrails.

        Fluxo:
            1. Input guardrail verifica prompt injection.
            2. Loop ReAct (Thought → Action → Observation) até end_turn ou max_iterations.
            3. Mensagens mantidas em formato normalizado (dicts) — agnóstico ao provider.
            4. Output guardrail sanitiza PII da resposta final.

        Args:
            query: Pergunta ou instrução do usuário em linguagem natural.

        Returns:
            AgentResponse com resposta sanitizada, iterações e ferramentas usadas.

        Raises:
            ValueError: Se a query for bloqueada pelo input guardrail.
        """
        guardrail_result = self.input_guardrail.check(query)
        if not guardrail_result["allowed"]:
            logger.warning(
                "Query bloqueada pelo input guardrail",
                extra={"reason": guardrail_result["reason"], "category": guardrail_result["category"]},
            )
            raise ValueError(
                f"Requisição bloqueada: {guardrail_result['reason']} (OWASP {guardrail_result['category']})"
            )

        # Histórico de mensagens em formato normalizado (dicts puros — agnóstico ao provider)
        messages: list[dict[str, Any]] = [{"role": "user", "content": query}]
        iterations: int = 0
        tools_used: list[str] = []

        while iterations < self.max_iterations:
            iterations += 1

            chat_resp = self.provider.chat(
                messages=messages,
                system=SYSTEM_PROMPT,
                tools=TOOLS_SCHEMA,
                max_tokens=2048,
            )

            logger.info(
                "Ciclo ReAct",
                extra={
                    "iteration": iterations,
                    "stop_reason": chat_resp.stop_reason,
                    "input_tokens": chat_resp.input_tokens,
                    "output_tokens": chat_resp.output_tokens,
                    "provider": self.provider.provider_name,
                },
            )

            if chat_resp.stop_reason == "end_turn":
                sanitized, had_pii = self.output_guardrail.apply(chat_resp.text)
                logger.info(
                    "Agente concluído",
                    extra={"iterations": iterations, "had_pii": had_pii, "tools_used": tools_used},
                )
                return AgentResponse(
                    answer=sanitized,
                    iterations=iterations,
                    had_pii=had_pii,
                    tools_used=tools_used,
                    provider=self.provider.provider_name,
                    model=self.provider.model_name,
                )

            # Serializar resposta do assistant em formato normalizado (dicts puros)
            assistant_content: list[dict[str, Any]] = []
            if chat_resp.text:
                assistant_content.append({"type": "text", "text": chat_resp.text})
            for tc in chat_resp.tool_calls:
                assistant_content.append({
                    "type": "tool_use",
                    "id": tc.id,
                    "name": tc.name,
                    "input": tc.input,
                })
            messages.append({"role": "assistant", "content": assistant_content})

            # Executar ferramentas e acumular resultados
            tool_results: list[dict[str, Any]] = []
            for tc in chat_resp.tool_calls:
                tools_used.append(tc.name)
                tool_output = self._execute_tool(tc.name, tc.input)
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tc.id,
                    "content": tool_output,
                })

            if tool_results:
                messages.append({"role": "user", "content": tool_results})

        logger.warning(
            "Limite de iterações atingido — loop interrompido (OWASP LLM08)",
            extra={"max_iterations": self.max_iterations, "tools_used": tools_used},
        )
        return AgentResponse(
            answer=(
                "Análise interrompida: limite de iterações atingido. "
                "Por favor, reformule a pergunta de forma mais específica."
            ),
            iterations=iterations,
            had_pii=False,
            tools_used=tools_used,
            provider=self.provider.provider_name,
            model=self.provider.model_name,
        )
