#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
source ./common_env.sh

if curl -sf "http://127.0.0.1:${ONLINE_PROXY_PORT}/v1/models" >/dev/null 2>&1; then
    echo "Port ${ONLINE_PROXY_PORT} already has a running proxy/service." >&2
    exit 1
fi

python3 ./03a_serial_vllm_proxy.py \
    --host 127.0.0.1 \
    --port "${ONLINE_PROXY_PORT}" \
    --upstream "${TARGET_VLLM_BASE_URL}" \
    --timeout 600
