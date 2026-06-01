#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────
#  download-models.sh
#  Downloads Qwen3.6-35B-A3B-NVFP4 to MODEL_PATH using hf CLI.
#  Run this once before starting the stack.
# ─────────────────────────────────────────────────────────────

set -euo pipefail

MODEL_PATH="${MODEL_PATH:-/data/models}"

echo "-> Installing huggingface_hub if not present..."
pip install -q "huggingface_hub[cli]"

echo "-> Model will be downloaded to: $MODEL_PATH"
mkdir -p "$MODEL_PATH"

download() {
    local repo="$1"
    local dest="$MODEL_PATH/$(basename "$repo")"
    if [ -d "$dest" ]; then
        echo "  Already present: $dest -- skipping."
    else
        echo "  Downloading $repo ..."
        hf download "$repo" \
            --local-dir "$dest" \
            ${HF_TOKEN:+--token "$HF_TOKEN"}
        echo "  Done: $dest"
    fi
}

download "nvidia/Qwen3.6-35B-A3B-NVFP4"

echo ""
echo "Models ready. You can now run: docker compose up -d"
