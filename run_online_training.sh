#!/bin/bash
# Start target vLLM, train online, and stop the target server on exit.
# Data preparation is intentionally separate and must be completed first.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "${SCRIPT_DIR}"
source ./common_env.sh

VLLM_LOG="${LOG_DIR}/target_vllm_$(date +%Y%m%d_%H%M%S).log"
bash ./03_launch_target_vllm.sh >"${VLLM_LOG}" 2>&1 &
VLLM_PID=$!

cleanup() {
    echo "Stopping target vLLM pid=${VLLM_PID}"
    kill -TERM "${VLLM_PID}" 2>/dev/null || true
    wait "${VLLM_PID}" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

echo "Target vLLM log: ${VLLM_LOG}"
bash ./04_train_online.sh 2>&1 | tee "${LOG_DIR}/train_online_$(date +%Y%m%d_%H%M%S).log"

