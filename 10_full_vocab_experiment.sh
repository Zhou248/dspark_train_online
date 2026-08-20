#!/bin/bash
# Train a DSpark experiment with the full Qwen3.6 target vocabulary.
set -euo pipefail
cd "$(dirname "$0")"

MODE="${1:-smoke}"
BASE_DIR="${FULL_VOCAB_BASE_DIR:-/home/z00909726/scripts/qwen36_dspark_online}"

# Force full-vocabulary training. 248320 / 32000 = 7.76, so reduce anchors
# proportionally from 512 to 64 to control logits/loss activation memory.
export DRAFT_VOCAB_SIZE=248320
export ONLINE_MAX_ANCHORS="${FULL_VOCAB_MAX_ANCHORS:-64}"
export EPOCHS="${FULL_VOCAB_EPOCHS:-3}"
export LOG_FREQ="${LOG_FREQ:-1}"

case "${MODE}" in
    smoke)
        export ONLINE_WORK_DIR="${FULL_VOCAB_SMOKE_DIR:-${BASE_DIR}/work_full_vocab_smoke}"
        export MAX_SAMPLES="${FULL_VOCAB_SMOKE_SAMPLES:-200}"
        export MAX_STEPS="${FULL_VOCAB_SMOKE_STEPS:-20}"
        echo "Full-vocab smoke: samples=${MAX_SAMPLES}, steps=${MAX_STEPS}"
        bash ./01_prepare_data.sh
        bash ./run_online_training.sh
        ;;
    prepare)
        export ONLINE_WORK_DIR="${FULL_VOCAB_WORK_DIR:-${BASE_DIR}/work_full_vocab_20k}"
        export MAX_SAMPLES="${FULL_VOCAB_MAX_SAMPLES:-20000}"
        echo "Preparing full-vocab formal run: samples=${MAX_SAMPLES}"
        bash ./01_prepare_data.sh
        ;;
    train)
        export ONLINE_WORK_DIR="${FULL_VOCAB_WORK_DIR:-${BASE_DIR}/work_full_vocab_20k}"
        unset MAX_STEPS
        echo "Full-vocab formal training: epochs=${EPOCHS}"
        bash ./run_online_training.sh
        ;;
    *)
        echo "Usage: bash 10_full_vocab_experiment.sh {smoke|prepare|train}" >&2
        exit 2
        ;;
esac
