#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
source ./common_env.sh

if [ ! -f "${PREPARED_DATA_DIR}/dataset_info.json" ]; then
    echo "Prepared dataset not found: ${PREPARED_DATA_DIR}" >&2
    echo "Run bash 01_prepare_data.sh first." >&2
    exit 1
fi
if [ ! -f "${DRAFT_CONFIG_DIR}/config.json" ]; then
    echo "Draft config not found: ${DRAFT_CONFIG_DIR}/config.json" >&2
    exit 1
fi

echo "Waiting for target vLLM at ${VLLM_ENDPOINT}"
for _ in $(seq 1 180); do
    if curl -sf "${VLLM_ENDPOINT}/models" >/dev/null 2>&1; then
        break
    fi
    sleep 2
done
if ! curl -sf "${VLLM_ENDPOINT}/models" >/dev/null 2>&1; then
    echo "Target vLLM did not become ready." >&2
    exit 1
fi

echo "Online training NPUs=${TRAIN_NPUS} nproc=${TRAIN_NPROC}"
echo "Training memory profile: FSDP_SHARD=${FSDP_SHARD} MAX_ANCHORS=${MAX_ANCHORS}"
echo "Generated hidden states are consumed immediately and deleted."
TRAIN_LIMIT_ARGS=()
if [ -n "${MAX_STEPS}" ]; then
    TRAIN_LIMIT_ARGS+=(--max-steps "${MAX_STEPS}")
    echo "Smoke-test limit: MAX_STEPS=${MAX_STEPS}"
fi
FSDP_ARGS=()
if [ "${FSDP_SHARD}" = "1" ]; then
    if [ "${TRAIN_NPROC}" -lt 2 ]; then
        echo "FSDP_SHARD=1 requires TRAIN_NPROC >= 2." >&2
        exit 1
    fi
    FSDP_ARGS+=(--fsdp-shard)
    echo "FSDP parameter/optimizer sharding enabled."
fi
cd "${MSPEC_ROOT}"
ASCEND_RT_VISIBLE_DEVICES="${TRAIN_NPUS}" torchrun \
    --standalone \
    --nproc_per_node "${TRAIN_NPROC}" \
    scripts/train.py \
    --verifier-name-or-path "${TARGET_MODEL}" \
    --draft-config "${DRAFT_CONFIG_DIR}" \
    --data-path "${PREPARED_DATA_DIR}" \
    --vllm-endpoint "${VLLM_ENDPOINT}" \
    --hidden-states-backend file \
    --hidden-states-path "${HIDDEN_STATES_DIR}" \
    --save-path "${CKPT_DIR}" \
    --speculator-type dspark \
    --block-size "${BLOCK_SIZE}" \
    --max-anchors "${MAX_ANCHORS}" \
    --target-layer-ids ${TARGET_LAYER_IDS} \
    --draft-vocab-size "${DRAFT_VOCAB_SIZE}" \
    --markov-rank "${MARKOV_RANK}" \
    --markov-head-type vanilla \
    --enable-confidence-head \
    --confidence-head-with-markov \
    --confidence-head-alpha 1.0 \
    --loss-fn '{"ce":0.1,"tv":0.9}' \
    --draft-attn-impl eager \
    --hidden-states-dtype bfloat16 \
    --total-seq-len "${SEQ_LENGTH}" \
    --train-data-ratio "${TRAIN_DATA_RATIO}" \
    --epochs "${EPOCHS}" \
    --lr "${LR}" \
    --optimizer adamw \
    --scheduler-type cosine \
    --scheduler-warmup-ratio 0.03 \
    --save-best \
    --checkpoint-freq 1 \
    --on-missing generate \
    --on-generate delete \
    --request-timeout 600 \
    --max-retries 3 \
    --num-workers 1 \
    --prefetch-factor 1 \
    --log-dir "${LOG_DIR}" \
    "${FSDP_ARGS[@]}" \
    "${TRAIN_LIMIT_ARGS[@]}"

echo "Training complete: ${CKPT_DIR}"
echo "Best checkpoint: ${CKPT_DIR}/checkpoint_best"
