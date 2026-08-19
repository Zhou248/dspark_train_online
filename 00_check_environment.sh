#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
source ./common_env.sh

failed=0
check_path() {
    local label="$1"
    local path="$2"
    if [ -e "${path}" ]; then
        echo "OK   ${label}: ${path}"
    else
        echo "FAIL ${label}: ${path}" >&2
        failed=1
    fi
}

check_path "msModelSpec" "${MSPEC_ROOT}/scripts/train.py"
check_path "target model" "${TARGET_MODEL}/config.json"
check_path "ALLaVA JSON" "${ALLAVA_JSON}"

for command_name in python3 torchrun curl; do
    if command -v "${command_name}" >/dev/null 2>&1; then
        echo "OK   command: ${command_name}"
    else
        echo "FAIL command: ${command_name}" >&2
        failed=1
    fi
done

python3 - <<'PY' || failed=1
import importlib

for name in (
    "torch",
    "torch_npu",
    "transformers",
    "datasets",
    "httpx",
    "openai",
    "safetensors",
    "PIL",
):
    module = importlib.import_module(name)
    print(f"OK   python: {name} {getattr(module, '__version__', '')}")
PY

python3 - <<'PY' || failed=1
import torch
import torch_npu  # noqa: F401

count = torch.npu.device_count()
print(f"visible NPU count: {count}")
if count < 16:
    raise SystemExit(f"Expected 16 NPUs before visibility filtering, found {count}")
PY

if [ "${failed}" -ne 0 ]; then
    echo "Environment check failed." >&2
    exit 1
fi
echo "Environment check passed."
