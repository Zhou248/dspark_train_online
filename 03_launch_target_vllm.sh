#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
source ./common_env.sh

if curl -sf "http://127.0.0.1:${VLLM_PORT}/health" >/dev/null 2>&1; then
    echo "Port ${VLLM_PORT} already has a healthy vLLM service." >&2
    exit 1
fi

echo "Starting target hidden-state server on NPUs ${TARGET_NPUS} (TP=${TARGET_TP})"
echo "Target endpoint: ${TARGET_VLLM_ENDPOINT}"
cd "${MSPEC_ROOT}"
ASCEND_RT_VISIBLE_DEVICES="${TARGET_NPUS}" python3 scripts/launch_vllm.py \
    "${TARGET_MODEL}" \
    --hidden-states-backend file \
    --hidden-states-path "${HIDDEN_STATES_DIR}" \
    --target-layer-ids ${TARGET_LAYER_IDS} \
    -- \
    --host 0.0.0.0 \
    --port "${VLLM_PORT}" \
    --tensor-parallel-size "${TARGET_TP}" \
    --seed 1024 \
    --max-num-seqs "${VLLM_MAX_NUM_SEQS}" \
    --max-model-len $((SEQ_LENGTH + 1024)) \
    --max-num-batched-tokens $((SEQ_LENGTH + 1024)) \
    --trust-remote-code \
    --gpu-memory-utilization 0.90 \
    --allowed-local-media-path /home \
    --limit-mm-per-prompt '{"image":1}' \
    --enforce-eager \
    --no-enable-prefix-caching \
    --no-async-scheduling \
    --additional-config '{"enable_cpu_binding":true}'
