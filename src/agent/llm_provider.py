"""Abstração de provider LLM: suporta Anthropic (cloud) e vLLM (local).

O agente ReAct usa esta interface para ser agnóstico ao backend de inferência.
Troca-se o provider via variável de ambiente LLM_PROVIDER sem alterar a lógica do agente.
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# ── Tipos normalizados ────────────────────────────────────────────────────────


@dataclass
class ToolCall:
    """Uma chamada de ferramenta retornada pelo LLM.

    Attributes:
        id: Identificador único da chamada (usado para correlacionar resultados).
        name: Nome da ferramenta conforme TOOLS_SCHEMA.
        input: Parâmetros de entrada como dict tipado.
    """

    id: str
    name: str
    input: dict[str, Any]


@dataclass
class ChatResponse:
    """Resposta normalizada do LLM, independente do provider.

    Attributes:
        stop_reason: Motivo de parada: 'end_turn', 'tool_use' ou 'max_tokens'.
        text: Texto gerado (resposta final quando stop_reason == 'end_turn').
        tool_calls: Lista de ferramentas a executar (quando stop_reason == 'tool_use').
        input_tokens: Tokens consumidos no prompt.
        output_tokens: Tokens gerados na resposta.
    """

    stop_reason: str
    text: str
    tool_calls: list[ToolCall]
    input_tokens: int
    output_tokens: int


# ── Interface base ────────────────────────────────────────────────────────────


class LLMProvider(ABC):
    """Interface abstrata para backends de inferência LLM."""

    @abstractmethod
    def chat(
        self,
        messages: list[dict[str, Any]],
        system: str,
        tools: list[dict[str, Any]],
        max_tokens: int = 2048,
    ) -> ChatResponse:
        """Executa uma chamada de chat com tool_use.

        Args:
            messages: Histórico de mensagens no formato normalizado (dicts).
            system: Prompt de sistema.
            tools: Schema das ferramentas no formato Anthropic (input_schema).
            max_tokens: Limite de tokens na resposta.

        Returns:
            ChatResponse normalizada com stop_reason, text, tool_calls e usage.
        """
        ...

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Identificador do modelo em uso."""
        ...

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Nome do provider (anthropic, vllm, etc.)."""
        ...


# ── Provider Anthropic ────────────────────────────────────────────────────────


class AnthropicProvider(LLMProvider):
    """Provider usando Anthropic SDK (Claude Haiku/Sonnet via API cloud).

    Usa tool_use nativo do SDK — sem conversão de formato.
    Fallback padrão quando LLM_PROVIDER=anthropic ou não configurado.
    """

    def __init__(self, model: str, api_key: str | None = None) -> None:
        """Inicializa o provider Anthropic.

        Args:
            model: Identificador do modelo (ex: 'claude-haiku-4-5-20251001').
            api_key: Chave da API. Lê ANTHROPIC_API_KEY se None.
        """
        try:
            import anthropic as _anthropic

            self._client = _anthropic.Anthropic(api_key=api_key)
        except ImportError as exc:
            raise ImportError("anthropic package necessário. Execute: pip install anthropic") from exc

        self._model = model
        logger.info("AnthropicProvider inicializado", extra={"model": model})

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def provider_name(self) -> str:
        return "anthropic"

    def chat(
        self,
        messages: list[dict[str, Any]],
        system: str,
        tools: list[dict[str, Any]],
        max_tokens: int = 2048,
    ) -> ChatResponse:
        # Anthropic aceita diretamente o formato normalizado (dicts com type/text/tool_use)
        response = self._client.messages.create(
            model=self._model,
            max_tokens=max_tokens,
            system=system,
            tools=tools,  # type: ignore[arg-type]
            messages=messages,  # type: ignore[arg-type]
        )

        text = ""
        tool_calls: list[ToolCall] = []

        for block in response.content:
            if hasattr(block, "text"):
                text = block.text
            elif getattr(block, "type", None) == "tool_use":
                tool_calls.append(
                    ToolCall(
                        id=block.id,
                        name=block.name,
                        input=dict(block.input),
                    )
                )

        stop_reason = "tool_use" if tool_calls else (response.stop_reason or "end_turn")

        return ChatResponse(
            stop_reason=stop_reason,
            text=text,
            tool_calls=tool_calls,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
        )


# ── Provider vLLM (OpenAI-compatible) ────────────────────────────────────────


class VLLMProvider(LLMProvider):
    """Provider usando vLLM via API OpenAI-compatible (LLM local com quantização).

    vLLM expõe endpoint idêntico ao OpenAI — usa openai.OpenAI apontando para
    o servidor local. Suporta Llama 3.1, Mistral, Qwen e outros HuggingFace models.

    Referência: https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html
    """

    def __init__(self, model: str, base_url: str) -> None:
        """Inicializa o provider vLLM.

        Args:
            model: Nome do modelo HuggingFace (ex: 'meta-llama/Meta-Llama-3.1-8B-Instruct').
            base_url: URL base do servidor vLLM (ex: 'http://localhost:8080/v1').
        """
        try:
            from openai import OpenAI

            # api_key não é validada pelo vLLM — qualquer string serve
            self._client = OpenAI(base_url=base_url, api_key="vllm-local-key")
        except ImportError as exc:
            raise ImportError("openai package necessário para VLLMProvider. Execute: pip install openai") from exc

        self._model = model
        self._base_url = base_url
        logger.info("VLLMProvider inicializado", extra={"model": model, "base_url": base_url})

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def provider_name(self) -> str:
        return "vllm"

    def _tools_to_openai(self, tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Converte schema Anthropic para formato OpenAI function calling.

        Anthropic:  {"name": ..., "description": ..., "input_schema": {...}}
        OpenAI:     {"type": "function", "function": {"name": ..., "description": ..., "parameters": {...}}}
        """
        return [
            {
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t.get("description", ""),
                    "parameters": t.get("input_schema", {"type": "object", "properties": {}}),
                },
            }
            for t in tools
        ]

    def _messages_to_openai(
        self,
        messages: list[dict[str, Any]],
        system: str,
    ) -> list[dict[str, Any]]:
        """Converte histórico de mensagens do formato normalizado para OpenAI.

        O formato normalizado usa dicts com 'type' (text, tool_use, tool_result),
        igual ao formato Anthropic. Aqui fazemos a tradução para OpenAI.
        """
        result: list[dict[str, Any]] = [{"role": "system", "content": system}]

        for msg in messages:
            role = msg["role"]
            content = msg["content"]

            if isinstance(content, str):
                result.append({"role": role, "content": content})
                continue

            if not isinstance(content, list):
                continue

            # Verificar se é lista de tool_result (mensagem de resultado de ferramentas)
            if all(isinstance(b, dict) and b.get("type") == "tool_result" for b in content):
                for block in content:
                    result.append({
                        "role": "tool",
                        "tool_call_id": block["tool_use_id"],
                        "content": block["content"],
                    })
                continue

            # Mensagem de assistant com texto e/ou tool_use
            text_parts: list[str] = []
            tool_calls_list: list[dict[str, Any]] = []

            for block in content:
                if isinstance(block, dict):
                    if block.get("type") == "text":
                        text_parts.append(block.get("text", ""))
                    elif block.get("type") == "tool_use":
                        tool_calls_list.append({
                            "id": block["id"],
                            "type": "function",
                            "function": {
                                "name": block["name"],
                                "arguments": json.dumps(block["input"], ensure_ascii=False),
                            },
                        })

            asst_msg: dict[str, Any] = {"role": "assistant"}
            if text_parts:
                asst_msg["content"] = " ".join(text_parts)
            if tool_calls_list:
                asst_msg["tool_calls"] = tool_calls_list
            if asst_msg.keys() - {"role"}:  # tem conteúdo além do role
                result.append(asst_msg)

        return result

    def chat(
        self,
        messages: list[dict[str, Any]],
        system: str,
        tools: list[dict[str, Any]],
        max_tokens: int = 2048,
    ) -> ChatResponse:
        openai_messages = self._messages_to_openai(messages, system)
        openai_tools = self._tools_to_openai(tools)

        response = self._client.chat.completions.create(
            model=self._model,
            messages=openai_messages,  # type: ignore[arg-type]
            tools=openai_tools,  # type: ignore[arg-type]
            tool_choice="auto",
            max_tokens=max_tokens,
            temperature=0.0,
        )

        choice = response.choices[0]
        message = choice.message

        text = message.content or ""
        tool_calls: list[ToolCall] = []

        if message.tool_calls:
            for tc in message.tool_calls:
                try:
                    tc_input = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    tc_input = {}
                tool_calls.append(ToolCall(id=tc.id, name=tc.function.name, input=tc_input))

        finish_reason = choice.finish_reason or "stop"
        stop_reason = "tool_use" if finish_reason == "tool_calls" else "end_turn"

        usage = response.usage
        return ChatResponse(
            stop_reason=stop_reason,
            text=text,
            tool_calls=tool_calls,
            input_tokens=usage.prompt_tokens if usage else 0,
            output_tokens=usage.completion_tokens if usage else 0,
        )


# ── Factory ───────────────────────────────────────────────────────────────────

_SUPPORTED_ANTHROPIC_MODELS = {
    "claude-haiku-4-5-20251001",
    "claude-sonnet-4-6",
    "claude-opus-4-7",
}

_DEFAULT_VLLM_MODEL = "meta-llama/Meta-Llama-3.1-8B-Instruct"
_DEFAULT_VLLM_URL = "http://localhost:8080/v1"


def create_provider(
    provider_name: str | None = None,
    model: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
) -> LLMProvider:
    """Cria um LLMProvider a partir de parâmetros ou variáveis de ambiente.

    Variáveis de ambiente reconhecidas:
        LLM_PROVIDER:    'anthropic' (padrão) ou 'vllm'
        ANTHROPIC_API_KEY: Chave Anthropic (obrigatória se provider=anthropic)
        AGENT_MODEL:      Modelo Anthropic (padrão: claude-haiku-4-5-20251001)
        VLLM_BASE_URL:    URL do servidor vLLM (padrão: http://localhost:8080/v1)
        VLLM_MODEL:       Modelo HuggingFace para vLLM

    Args:
        provider_name: 'anthropic' ou 'vllm'. Sobrescreve LLM_PROVIDER.
        model: Identificador do modelo. Sobrescreve AGENT_MODEL / VLLM_MODEL.
        api_key: Chave Anthropic. Sobrescreve ANTHROPIC_API_KEY.
        base_url: URL base vLLM. Sobrescreve VLLM_BASE_URL.

    Returns:
        Instância de LLMProvider configurada.

    Raises:
        ValueError: Se o provider_name não for reconhecido.
    """
    resolved_provider = provider_name or os.getenv("LLM_PROVIDER", "anthropic")

    if resolved_provider == "vllm":
        resolved_model = model or os.getenv("VLLM_MODEL", _DEFAULT_VLLM_MODEL)
        resolved_url = base_url or os.getenv("VLLM_BASE_URL", _DEFAULT_VLLM_URL)
        logger.info(
            "Criando VLLMProvider",
            extra={"model": resolved_model, "base_url": resolved_url},
        )
        return VLLMProvider(model=resolved_model, base_url=resolved_url)

    if resolved_provider == "anthropic":
        resolved_model = model or os.getenv("AGENT_MODEL", "claude-haiku-4-5-20251001")
        resolved_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        logger.info("Criando AnthropicProvider", extra={"model": resolved_model})
        return AnthropicProvider(model=resolved_model, api_key=resolved_key)

    raise ValueError(
        f"Provider desconhecido: '{resolved_provider}'. "
        f"Use 'anthropic' ou 'vllm' via LLM_PROVIDER."
    )
