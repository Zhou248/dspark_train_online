#!/bin/bash
# Send one text request and one image-understanding request to DSpark vLLM.
set -euo pipefail
cd "$(dirname "$0")"
source ./common_env.sh

SERVE_PORT="${SERVE_PORT:-8100}"
API_BASE="${API_BASE:-http://127.0.0.1:${SERVE_PORT}/v1}"
MODEL_NAME="${MODEL_NAME:-${TARGET_MODEL}}"
IMAGE_PATH="${1:-${IMAGE_PATH:-}}"
TEXT_PROMPT="${TEXT_PROMPT:-请用三句话介绍投机推理的工作原理。}"
IMAGE_PROMPT="${IMAGE_PROMPT:-请详细描述这张图片中的主要内容。}"
MAX_TOKENS="${MAX_TOKENS:-256}"

if [ -z "${IMAGE_PATH}" ]; then
    echo "Usage: bash 08_curl_requests.sh /absolute/path/to/image.jpg" >&2
    echo "Or set IMAGE_PATH=/absolute/path/to/image.jpg" >&2
    exit 2
fi
if [ ! -f "${IMAGE_PATH}" ]; then
    echo "Image not found: ${IMAGE_PATH}" >&2
    exit 2
fi
if ! [[ "${MAX_TOKENS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "MAX_TOKENS must be a positive integer: ${MAX_TOKENS}" >&2
    exit 2
fi

echo "Checking vLLM endpoint: ${API_BASE}"
curl -fsS "${API_BASE}/models" >/dev/null

echo
echo "========== 1. Pure text request =========="
python3 - "${MODEL_NAME}" "${TEXT_PROMPT}" "${MAX_TOKENS}" <<'PY' |
import json
import sys

model, prompt, max_tokens = sys.argv[1:]
json.dump(
    {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": int(max_tokens),
    },
    sys.stdout,
    ensure_ascii=False,
)
PY
    curl -fsS --show-error \
        -H "Authorization: Bearer EMPTY" \
        -H "Content-Type: application/json" \
        --data-binary @- \
        "${API_BASE}/chat/completions" |
    python3 -m json.tool --no-ensure-ascii

echo
echo "========== 2. Multimodal image request =========="
echo "Image: ${IMAGE_PATH}"
python3 - "${MODEL_NAME}" "${IMAGE_PATH}" "${IMAGE_PROMPT}" "${MAX_TOKENS}" <<'PY' |
import base64
import json
import mimetypes
import sys
from pathlib import Path

model, image_name, prompt, max_tokens = sys.argv[1:]
image_path = Path(image_name)
mime_type = mimetypes.guess_type(image_path.name)[0] or "image/jpeg"
image_base64 = base64.b64encode(image_path.read_bytes()).decode("ascii")
json.dump(
    {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{mime_type};base64,{image_base64}"
                        },
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ],
        "temperature": 0,
        "max_tokens": int(max_tokens),
    },
    sys.stdout,
    ensure_ascii=False,
)
PY
    curl -fsS --show-error \
        -H "Authorization: Bearer EMPTY" \
        -H "Content-Type: application/json" \
        --data-binary @- \
        "${API_BASE}/chat/completions" |
    python3 -m json.tool --no-ensure-ascii

echo
echo "Both requests completed."
