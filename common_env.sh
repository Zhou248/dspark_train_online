#!/bin/bash
# Qwen3.6-35B-A3B + ALLaVA-4V + DSpark online training configuration.
# Override any value before launching a script, for example:
#   MAX_SAMPLES=20000 bash 01_prepare_data.sh

set -euo pipefail

export MSPEC_ROOT="${MSPEC_ROOT:-/home/z00909726/msModelSpec-Dev}"
export TARGET_MODEL="${TARGET_MODEL:-/home/z00909726/weights/Qwen3.6-35B-A3B}"
export ALLAVA_ROOT="${ALLAVA_ROOT:-/home/w00608002/models/ALLaVA-4V/allava_laion}"
export ALLAVA_JSON="${ALLAVA_JSON:-${ALLAVA_ROOT}/ALLaVA-Instruct-LAION-4V.mm.json}"
# Use ONLINE_* overrides so variables exported by the older offline workflow
# cannot redirect this project into qwen36_dspark/work by accident.
export ONLINE_WORK_DIR="${ONLINE_WORK_DIR:-/home/z00909726/scripts/qwen36_dspark_online/work}"
export WORK_DIR="${ONLINE_WORK_DIR}"
export NORMALIZED_DATA="${ONLINE_NORMALIZED_DATA:-${WORK_DIR}/dataset/conversations.jsonl}"
export PREPARED_DATA_DIR="${ONLINE_PREPARED_DATA_DIR:-${WORK_DIR}/prepared}"
export DRAFT_CONFIG_DIR="${ONLINE_DRAFT_CONFIG_DIR:-${WORK_DIR}/draft_config}"
export HIDDEN_STATES_DIR="${ONLINE_HIDDEN_STATES_DIR:-${WORK_DIR}/online_hidden_states}"
export CKPT_DIR="${ONLINE_CKPT_DIR:-${WORK_DIR}/checkpoints}"
export LOG_DIR="${ONLINE_LOG_DIR:-${WORK_DIR}/logs}"

# Start with 5k for an end-to-end run. A useful domain drafter normally needs
# 20k-50k+ high-quality samples; raise this only after the smoke run is stable.
export MAX_SAMPLES="${MAX_SAMPLES:-5000}"
export SEQ_LENGTH="${SEQ_LENGTH:-4096}"
export PREPROCESS_WORKERS="${PREPROCESS_WORKERS:-8}"
export MINIMUM_VALID_TOKENS="${MINIMUM_VALID_TOKENS:-16}"
# Cap source images before prepare_data. At ~1M pixels Qwen's visual token count
# leaves ample room for text inside SEQ_LENGTH=4096.
export MAX_IMAGE_PIXELS="${MAX_IMAGE_PIXELS:-1048576}"
export MAX_IMAGE_SIDE="${MAX_IMAGE_SIDE:-2048}"
export RESIZED_IMAGE_DIR="${ONLINE_RESIZED_IMAGE_DIR:-${WORK_DIR}/dataset/resized_images}"

# Target hidden-state extraction layers. launch_vllm.py appends final layer 40.
export TARGET_LAYER_IDS="${TARGET_LAYER_IDS:-2 20 37}"
export NUM_DRAFT_LAYERS="${NUM_DRAFT_LAYERS:-3}"
export DRAFT_VOCAB_SIZE="${DRAFT_VOCAB_SIZE:-32000}"
export BLOCK_SIZE="${BLOCK_SIZE:-7}"
# Eager DSpark attention at 3072 anchors requested 32.81 GiB for one softmax
# tensor on A2. 512 keeps the same training objective with a smaller anchor
# sample per packed batch and is the safe starting point for 4096-token data.
export MAX_ANCHORS="${ONLINE_MAX_ANCHORS:-512}"
export MARKOV_RANK="${MARKOV_RANK:-32}"
export EPOCHS="${EPOCHS:-5}"
export LR="${LR:-3e-4}"
export TRAIN_DATA_RATIO="${TRAIN_DATA_RATIO:-0.95}"
export LOG_FREQ="${LOG_FREQ:-1}"
export MAX_STEPS="${MAX_STEPS:-}"

# Target uses cards 0-3. Training uses cards 8-15 with FSDP so parameters and
# optimizer state are sharded instead of replicated by DDP.
export TARGET_NPUS="${TARGET_NPUS:-0,1,2,3}"
export TARGET_TP="${TARGET_TP:-4}"
export TRAIN_NPUS="${ONLINE_TRAIN_NPUS:-8,9,10,11,12,13,14,15}"
export TRAIN_NPROC="${ONLINE_TRAIN_NPROC:-8}"
export FSDP_SHARD="${ONLINE_FSDP_SHARD:-1}"
export VLLM_PORT="${VLLM_PORT:-8000}"
export ONLINE_PROXY_PORT="${ONLINE_PROXY_PORT:-8001}"
export TARGET_VLLM_BASE_URL="http://127.0.0.1:${VLLM_PORT}"
export TARGET_VLLM_ENDPOINT="${TARGET_VLLM_BASE_URL}/v1"
export VLLM_ENDPOINT="http://127.0.0.1:${ONLINE_PROXY_PORT}/v1"
export VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-1}"

export ASCEND_TOOLKIT_HOME="${ASCEND_TOOLKIT_HOME:-/usr/local/Ascend/ascend-toolkit/latest}"
ASCEND_LIBRARY_PATH="${ASCEND_TOOLKIT_HOME}/lib64:${ASCEND_TOOLKIT_HOME}/runtime/lib64"
ASCEND_LIBRARY_PATH="${ASCEND_LIBRARY_PATH}:/usr/local/Ascend/driver/lib64/common"
ASCEND_LIBRARY_PATH="${ASCEND_LIBRARY_PATH}:/usr/local/Ascend/driver/lib64/driver"
export LD_LIBRARY_PATH="${ASCEND_LIBRARY_PATH}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export PYTHONPATH="${MSPEC_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export PYTORCH_NPU_ALLOC_CONF="${PYTORCH_NPU_ALLOC_CONF:-expandable_segments:True}"
export HCCL_BUFFSIZE="${HCCL_BUFFSIZE:-1024}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export TASK_QUEUE_ENABLE="${TASK_QUEUE_ENABLE:-0}"
unset HCCL_OP_EXPANSION_MODE || true

mkdir -p "$(dirname "${NORMALIZED_DATA}")" "${PREPARED_DATA_DIR}" \
    "${DRAFT_CONFIG_DIR}" "${HIDDEN_STATES_DIR}" "${CKPT_DIR}" "${LOG_DIR}" \
    "${RESIZED_IMAGE_DIR}"
