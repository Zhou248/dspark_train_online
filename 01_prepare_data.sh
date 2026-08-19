#!/bin/bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "${SCRIPT_DIR}"
source ./common_env.sh

echo "[1/4] Normalizing ALLaVA-4V"
NORMALIZE_ARGS=()
if [ "${SKIP_INVALID_SOURCE_ROWS:-0}" = "1" ]; then
    NORMALIZE_ARGS+=(--skip-invalid)
fi
python3 ./01_normalize_allava.py \
    --input "${ALLAVA_JSON}" \
    --output "${NORMALIZED_DATA}" \
    --max-samples "${MAX_SAMPLES}" \
    "${NORMALIZE_ARGS[@]}"

echo "[2/4] Preparing tokens, loss masks, multimodal messages and token frequencies"
cd "${MSPEC_ROOT}"
python3 scripts/prepare_data.py \
    --model "${TARGET_MODEL}" \
    --data "${NORMALIZED_DATA}" \
    --output "${PREPARED_DATA_DIR}" \
    --max-samples "${MAX_SAMPLES}" \
    --seq-length "${SEQ_LENGTH}" \
    --minimum-valid-tokens "${MINIMUM_VALID_TOKENS}" \
    --num-preprocessing-workers "${PREPROCESS_WORKERS}" \
    --overwrite

echo "[3/4] Building the 1-D Qwen3 DSpark decoder config"
python3 "${SCRIPT_DIR}/02_build_draft_config.py" \
    --target-model "${TARGET_MODEL}" \
    --output-dir "${DRAFT_CONFIG_DIR}" \
    --num-layers "${NUM_DRAFT_LAYERS}"

echo "[4/4] Auditing prepared data"
python3 "${SCRIPT_DIR}/02_inspect_prepared_data.py" \
    --prepared-data "${PREPARED_DATA_DIR}" \
    --seq-length "${SEQ_LENGTH}"

echo "Prepared data: ${PREPARED_DATA_DIR}"
echo "Draft config: ${DRAFT_CONFIG_DIR}"

