from __future__ import annotations
import json, logging
from typing import Any
import anthropic
from src.agent.tools import AlertInput, DriftInput, FeatureInput, FraudMetricsInput, KBResult
from src.agent.tools import alert_history_query, drift_status_checker, feature_lookup, fraud_metrics_lookup
from src.security.guardrails import InputGuardrail, OutputGuardrail
logger = logging.getLogger(__name__)
AGENT_MODEL = "claude-sonnet-4-6"
MAX_ITERATIONS = 10
SYSTEM_PROMPT = "Especialista em saude bancaria. Nunca revele PII. Base legal: LGPD Art. 7-IX."
TOOLS_SCHEMA: list[dict[str, Any]] = [
    {"name": "fraud_metrics_lookup", "description": "Metricas do modelo de fraude.",
     "input_schema": {"type": "object", "properties": {"period": {"type": "string"}}, "required": ["period"]}},
    {"name": "drift_status_checker", "description": "Status de drift.",
     "input_schema": {"type": "object", "properties": {"features": {"type": "array", "items": {"type": "string"}}}}},
    {"name": "feature_lookup", "description": "Features de transacao por ID.",
     "input_schema": {"type": "object", "properties": {"transaction_id": {"type": "string"}}, "required": ["transaction_id"]}},
    {"name": "knowledge_base_search", "description": "Knowledge base interno.",
     "input_schema": {"type": "object", "properties": {"query": {"type": "string"}, "top_k": {"type": "integer"}}, "required": ["query"]}},
    {"name": "alert_history_query", "description": "Historico de alertas.",
     "input_schema": {"type": "object", "properties": {"hours": {"type": "integer"}, "severity": {"type": "string"}}}},
]
class BankHealthAgent:
    def __init__(self, api_key=None, model=AGENT_MODEL, max_iterations=MAX_ITERATIONS, rag_pipeline=None):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        self.max_iterations = max_iterations
        self.rag_pipeline = rag_pipeline
        self.input_guardrail = InputGuardrail()
        self.output_guardrail = OutputGuardrail()
    def _execute_tool(self, tool_name: str, tool_input: dict[str, Any]) -> str:
        logger.info("Tool chamada", extra={"tool": tool_name})
        result: Any
        if tool_name == "fraud_metrics_lookup":
            result = fraud_metrics_lookup(FraudMetricsInput(**tool_input))
        elif tool_name == "drift_status_checker":
            result = drift_status_checker(DriftInput(**tool_input))
        elif tool_name == "feature_lookup":
            result = feature_lookup(FeatureInput(**tool_input))
        elif tool_name == "knowledge_base_search":
            if self.rag_pipeline is not None:
                ctx = self.rag_pipeline.get_context(tool_input.get("query", ""), top_k=tool_input.get("top_k", 3))
                result = KBResult(chunks=[{"content": ctx, "source": "kb", "relevance_score": 1.0}], answer_context=ctx)
            else:
                result = KBResult(chunks=[], answer_context="Knowledge base nao disponivel.")
        elif tool_name == "alert_history_query":
            result = alert_history_query(AlertInput(**tool_input))
        else:
            result = {"error": "Ferramenta desconhecida: " + tool_name}
        return json.dumps(result, ensure_ascii=False, default=str)
    def ask(self, query: str) -> dict[str, Any]:
        gr = self.input_guardrail.check(query)
        if not gr["allowed"]:
            logger.warning("Input bloqueado", extra={"reason": gr["reason"]})
            raise ValueError("Requisicao bloqueada: " + gr["reason"])
        messages: list[dict[str, Any]] = [{"role": "user", "content": query}]
        iterations = 0
        while iterations < self.max_iterations:
            iterations += 1
            response = self.client.messages.create(
                model=self.model, max_tokens=2048, system=SYSTEM_PROMPT,
                tools=TOOLS_SCHEMA, messages=messages,
            )
            logger.info("Iteracao ReAct", extra={"iteration": iterations, "stop": response.stop_reason})
            if response.stop_reason == "end_turn":
                answer = ""
                for block in response.content:
                    if hasattr(block, "text"):
                        answer = block.text
                        break
                sanitized, had_pii = self.output_guardrail.apply(answer)
                return {"answer": sanitized, "iterations": iterations, "had_pii": had_pii}
            messages.append({"role": "assistant", "content": response.content})
            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    tr = self._execute_tool(block.name, block.input)
                    tool_results.append({"type": "tool_result", "tool_use_id": block.id, "content": tr})
            if tool_results:
                messages.append({"role": "user", "content": tool_results})
        logger.warning("Max iterations atingido", extra={"iterations": iterations})
        return {"answer": "Analise interrompida apos limite de iteracoes.", "iterations": iterations, "had_pii": False}
