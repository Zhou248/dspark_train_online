#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
source ./common_env.sh

BEST_CKPT="${CKPT_DIR}/checkpoint_best"
SERVE_PORT="${SERVE_PORT:-8100}"
if [ ! -e "${BEST_CKPT}" ]; then
    echo "Best checkpoint not found: ${BEST_CKPT}" >&2
    exit 1
fi

echo "Serving target + DSpark on port ${SERVE_PORT}"
ASCEND_RT_VISIBLE_DEVICES="${TARGET_NPUS}" python3 -m vllm.entrypoints.cli.main serve \
    "${TARGET_MODEL}" \
    --host 0.0.0.0 \
    --port "${SERVE_PORT}" \
    --tensor-parallel-size "${TARGET_TP}" \
    --trust-remote-code \
    --max-model-len $((SEQ_LENGTH + 1024)) \
    --max-num-seqs 1 \
    --gpu-memory-utilization 0.85 \
    --allowed-local-media-path /home \
    --limit-mm-per-prompt '{"image":1}' \
    --enforce-eager \
    --no-enable-prefix-caching \
    --no-async-scheduling \
    --speculative-config "{
        \"method\":\"dspark\",
        \"model\":\"${BEST_CKPT}\",
        \"num_speculative_tokens\":${BLOCK_SIZE},
        \"draft_sample_method\":\"greedy\"
    }"

