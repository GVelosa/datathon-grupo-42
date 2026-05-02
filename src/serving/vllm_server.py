"""Configuração e utilitários do servidor vLLM para LLM local com quantização.

O vLLM expõe uma API 100% compatível com OpenAI — o agente ReAct não percebe
diferença entre chamar Claude ou um Llama local. A quantização reduz uso de
memória GPU sem sacrifício crítico de qualidade para tarefas de raciocínio bancário.

Referência: https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# ── Modelos suportados ────────────────────────────────────────────────────────

SUPPORTED_MODELS: dict[str, dict[str, str]] = {
    # Llama 3.1 — excelente suporte multilíngue (PT-BR), padrão recomendado
    "llama-3.1-8b": {
        "hf_id": "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "description": "Llama 3.1 8B Instruct — melhor equilíbrio qualidade/velocidade",
        "vram_fp16_gb": "16",
        "vram_int8_gb": "10",
        "vram_int4_gb": "6",
    },
    # Qwen 2.5 — superior em português entre modelos 7B, alternativa recomendada
    "qwen2.5-7b": {
        "hf_id": "Qwen/Qwen2.5-7B-Instruct",
        "description": "Qwen 2.5 7B — melhor desempenho em PT-BR entre 7B",
        "vram_fp16_gb": "14",
        "vram_int8_gb": "8",
        "vram_int4_gb": "5",
    },
    # Mistral 7B — opção rápida para hardware limitado
    "mistral-7b": {
        "hf_id": "mistralai/Mistral-7B-Instruct-v0.3",
        "description": "Mistral 7B Instruct v0.3 — latência mínima",
        "vram_fp16_gb": "14",
        "vram_int8_gb": "8",
        "vram_int4_gb": "5",
    },
}

# ── Opções de quantização ─────────────────────────────────────────────────────

QUANTIZATION_OPTIONS: dict[str, dict[str, str]] = {
    "awq": {
        "flag": "--quantization awq",
        "bits": "4-bit",
        "description": "AWQ (Activation-aware Weight Quantization) — qualidade superior em 4-bit",
        "vram_reduction": "~60% vs FP16",
        "quality_impact": "< 2% degradação em benchmarks de raciocínio",
    },
    "int8": {
        "flag": "--quantization bitsandbytes --load-format bitsandbytes",
        "bits": "8-bit",
        "description": "INT8 via bitsandbytes — equilíbrio qualidade/memória",
        "vram_reduction": "~40% vs FP16",
        "quality_impact": "< 1% degradação em benchmarks de raciocínio",
    },
    "fp16": {
        "flag": "",
        "bits": "16-bit",
        "description": "FP16 nativo — qualidade máxima, requer mais VRAM",
        "vram_reduction": "0% (baseline)",
        "quality_impact": "0% (referência)",
    },
}

# ── Configuração ──────────────────────────────────────────────────────────────


@dataclass
class VLLMConfig:
    """Configuração completa do servidor vLLM.

    Attributes:
        model: Identificador HuggingFace do modelo (ex: 'meta-llama/Meta-Llama-3.1-8B-Instruct').
        quantization: Tipo de quantização: 'awq', 'int8' ou 'fp16'.
        port: Porta de serving (padrão: 8080 para não conflitar com fraud-api em 8000).
        max_model_len: Tamanho máximo do contexto em tokens.
        gpu_memory_utilization: Fração da VRAM a reservar para o modelo (0.0–1.0).
        dtype: Tipo de dado (auto detecta melhor opção para o hardware).
        tensor_parallel_size: Número de GPUs para tensor parallelism.
        host: Interface de binding do servidor.
        hf_token: Token HuggingFace para modelos com gate (ex: Llama).
    """

    model: str = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    quantization: str = "awq"
    port: int = 8080
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.90
    dtype: str = "auto"
    tensor_parallel_size: int = 1
    host: str = "0.0.0.0"
    hf_token: str = field(default_factory=lambda: os.getenv("HF_TOKEN", ""))

    @classmethod
    def from_env(cls) -> "VLLMConfig":
        """Cria configuração a partir de variáveis de ambiente."""
        return cls(
            model=os.getenv("VLLM_MODEL", "meta-llama/Meta-Llama-3.1-8B-Instruct"),
            quantization=os.getenv("VLLM_QUANTIZATION", "awq"),
            port=int(os.getenv("VLLM_PORT", "8080")),
            max_model_len=int(os.getenv("VLLM_MAX_MODEL_LEN", "4096")),
            gpu_memory_utilization=float(os.getenv("VLLM_GPU_MEMORY_UTIL", "0.90")),
            dtype=os.getenv("VLLM_DTYPE", "auto"),
            tensor_parallel_size=int(os.getenv("VLLM_TENSOR_PARALLEL", "1")),
        )

    def to_cli_args(self) -> list[str]:
        """Gera lista de argumentos CLI para o servidor vLLM.

        Returns:
            Lista de strings para passar ao subprocesso ou ao ENTRYPOINT do Docker.
        """
        args = [
            "--model", self.model,
            "--host", self.host,
            "--port", str(self.port),
            "--max-model-len", str(self.max_model_len),
            "--gpu-memory-utilization", str(self.gpu_memory_utilization),
            "--dtype", self.dtype,
            "--tensor-parallel-size", str(self.tensor_parallel_size),
            "--served-model-name", self.model.split("/")[-1],  # nome curto para logs
        ]

        quant_info = QUANTIZATION_OPTIONS.get(self.quantization, {})
        quant_flag = quant_info.get("flag", "")
        if quant_flag:
            # Separa flags compostas (ex: "--quantization bitsandbytes --load-format bitsandbytes")
            args.extend(quant_flag.split())

        if self.hf_token:
            args.extend(["--download-dir", "/root/.cache/huggingface"])

        return args

    def to_docker_command(self) -> str:
        """Gera o comando docker run completo para referência na documentação."""
        hf_env = f'--env "HF_TOKEN={self.hf_token}" \\' if self.hf_token else ""
        args_str = " \\\n  ".join(self.to_cli_args())

        return (
            f"docker run --runtime nvidia --gpus all \\\n"
            f"  -v ~/.cache/huggingface:/root/.cache/huggingface \\\n"
            f"  {hf_env}\n"
            f"  -p {self.port}:{self.port} \\\n"
            f"  --ipc=host \\\n"
            f"  vllm/vllm-openai:latest \\\n"
            f"  {args_str}"
        )


# ── Health check ──────────────────────────────────────────────────────────────


def check_vllm_health(base_url: str | None = None, timeout: float = 5.0) -> dict[str, object]:
    """Verifica se o servidor vLLM está respondendo.

    Args:
        base_url: URL base do servidor (padrão: VLLM_BASE_URL env var).
        timeout: Timeout em segundos para a requisição.

    Returns:
        Dict com 'healthy' (bool), 'model' (str) e 'error' (str|None).
    """
    import urllib.error
    import urllib.request

    url = base_url or os.getenv("VLLM_BASE_URL", "http://localhost:8080/v1")
    health_url = url.rstrip("/v1").rstrip("/") + "/health"

    try:
        with urllib.request.urlopen(health_url, timeout=timeout) as resp:
            return {"healthy": resp.status == 200, "url": health_url, "error": None}
    except urllib.error.URLError as exc:
        return {"healthy": False, "url": health_url, "error": str(exc)}
    except Exception as exc:
        return {"healthy": False, "url": health_url, "error": str(exc)}


def list_vllm_models(base_url: str | None = None, timeout: float = 5.0) -> list[str]:
    """Lista os modelos disponíveis no servidor vLLM.

    Args:
        base_url: URL base do servidor.
        timeout: Timeout em segundos.

    Returns:
        Lista de nomes de modelos ou lista vazia se servidor indisponível.
    """
    import json as _json
    import urllib.request

    url = base_url or os.getenv("VLLM_BASE_URL", "http://localhost:8080/v1")
    models_url = url.rstrip("/") + "/models"

    try:
        with urllib.request.urlopen(models_url, timeout=timeout) as resp:
            data = _json.loads(resp.read())
            return [m["id"] for m in data.get("data", [])]
    except Exception:
        return []


# ── Configurações de benchmark predefinidas ───────────────────────────────────

BENCHMARK_CONFIGS = {
    "vllm-llama-awq": VLLMConfig(
        model="meta-llama/Meta-Llama-3.1-8B-Instruct",
        quantization="awq",
        port=8080,
        max_model_len=4096,
        gpu_memory_utilization=0.90,
    ),
    "vllm-llama-int8": VLLMConfig(
        model="meta-llama/Meta-Llama-3.1-8B-Instruct",
        quantization="int8",
        port=8080,
        max_model_len=4096,
        gpu_memory_utilization=0.85,
    ),
    "vllm-qwen-awq": VLLMConfig(
        model="Qwen/Qwen2.5-7B-Instruct",
        quantization="awq",
        port=8080,
        max_model_len=4096,
        gpu_memory_utilization=0.90,
    ),
}


if __name__ == "__main__":
    # Exibe configuração atual e comando docker de referência
    import sys

    config = VLLMConfig.from_env()
    print("=== Configuração vLLM ===")
    print(f"Modelo:          {config.model}")
    print(f"Quantização:     {config.quantization} ({QUANTIZATION_OPTIONS.get(config.quantization, {}).get('bits', 'n/a')})")
    print(f"Porta:           {config.port}")
    print(f"Max context:     {config.max_model_len} tokens")
    print(f"GPU utilization: {config.gpu_memory_utilization:.0%}")
    print()
    print("=== Comando Docker (referência) ===")
    print(config.to_docker_command())
    print()
    print("=== Health check ===")
    health = check_vllm_health()
    status = "✓ Online" if health["healthy"] else f"✗ Offline ({health['error']})"
    print(f"Status: {status}")
    sys.exit(0 if health["healthy"] else 1)
