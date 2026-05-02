#!/bin/bash
# Entrypoint do container vLLM — monta argumentos a partir de env vars
# e inicia o servidor OpenAI-compatible.
#
# Variáveis de ambiente lidas:
#   VLLM_MODEL, VLLM_QUANTIZATION, VLLM_PORT, VLLM_MAX_MODEL_LEN,
#   VLLM_GPU_MEM_UTIL, VLLM_DTYPE, VLLM_TENSOR_PARALLEL, HF_TOKEN

set -e

VLLM_MODEL="${VLLM_MODEL:-meta-llama/Meta-Llama-3.1-8B-Instruct}"
VLLM_QUANTIZATION="${VLLM_QUANTIZATION:-awq}"
VLLM_PORT="${VLLM_PORT:-8080}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-4096}"
VLLM_GPU_MEM_UTIL="${VLLM_GPU_MEM_UTIL:-0.90}"
VLLM_DTYPE="${VLLM_DTYPE:-auto}"
VLLM_TENSOR_PARALLEL="${VLLM_TENSOR_PARALLEL:-1}"

echo "[vLLM] Iniciando servidor..."
echo "[vLLM] Modelo:          ${VLLM_MODEL}"
echo "[vLLM] Quantização:     ${VLLM_QUANTIZATION}"
echo "[vLLM] Porta:           ${VLLM_PORT}"
echo "[vLLM] Max context:     ${VLLM_MAX_MODEL_LEN} tokens"
echo "[vLLM] GPU utilization: ${VLLM_GPU_MEM_UTIL}"

# Monta os argumentos de quantização
QUANT_ARGS=""
if [ "${VLLM_QUANTIZATION}" = "awq" ]; then
    QUANT_ARGS="--quantization awq"
elif [ "${VLLM_QUANTIZATION}" = "int8" ]; then
    QUANT_ARGS="--quantization bitsandbytes --load-format bitsandbytes"
fi
# fp16: sem flags extras

exec python -m vllm.entrypoints.openai.api_server \
    --model "${VLLM_MODEL}" \
    --host 0.0.0.0 \
    --port "${VLLM_PORT}" \
    --max-model-len "${VLLM_MAX_MODEL_LEN}" \
    --gpu-memory-utilization "${VLLM_GPU_MEM_UTIL}" \
    --dtype "${VLLM_DTYPE}" \
    --tensor-parallel-size "${VLLM_TENSOR_PARALLEL}" \
    --served-model-name "$(basename ${VLLM_MODEL})" \
    ${QUANT_ARGS} \
    "$@"
