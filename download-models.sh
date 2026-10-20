#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────
#  download-models.sh
#  Downloads Qwen3.6-35B-A3B-NVFP4 and its DFlash draft model to MODEL_PATH.
#  Run this once before starting the stack.
#
#  BOTH are required: docker-compose.yml enables speculative decoding via
#  --speculative-config, which points at the DFlash draft model. Without it
#  vLLM will fail to start. To run without speculative decoding (~30% slower
#  at concurrency 1), remove the --speculative-config flag from docker-compose.yml.
#
#  Reads MODEL_PATH (and optional HF_TOKEN) from .env next to this script;
#  values already set in your shell take precedence. Safe to re-run: an
#  interrupted download resumes, and completed files are skipped.
#
#  Uses the `hf` CLI if it is on your PATH; otherwise runs it in a throwaway
#  python:3.12-slim container, so nothing is pip-installed on the host (DGX OS's
#  system Python is "externally managed" and refuses `pip install`).
# ─────────────────────────────────────────────────────────────

set -euo pipefail

cd "$(dirname "$0")"

# Load .env without overriding variables already exported in the shell.
if [ -f .env ]; then
    while IFS='=' read -r key value; do
        case "$key" in ''|\#*) continue ;; esac
        if [ -z "${!key:-}" ]; then export "$key=$value"; fi
    done < .env
fi

MODEL_PATH="${MODEL_PATH:-/data/models}"
HF_TOKEN="${HF_TOKEN:-}"

if ! mkdir -p "$MODEL_PATH" 2>/dev/null || [ ! -w "$MODEL_PATH" ]; then
    echo "Cannot write to $MODEL_PATH. Create it and take ownership first:" >&2
    echo "  sudo mkdir -p $MODEL_PATH && sudo chown \$USER:\$USER $MODEL_PATH" >&2
    exit 1
fi
MODEL_PATH="$(cd "$MODEL_PATH" && pwd)"   # absolute, for the docker bind mount
echo "-> Downloading to: $MODEL_PATH"

hf_download() {   # <repo> <dest-dir-name>
    if command -v hf >/dev/null 2>&1; then
        hf download "$1" --local-dir "$MODEL_PATH/$2" ${HF_TOKEN:+--token "$HF_TOKEN"}
    else
        docker run --rm \
            -v "$MODEL_PATH:/models" \
            -e HF_TOKEN="$HF_TOKEN" \
            python:3.12-slim \
            sh -c "pip install -q --root-user-action=ignore --disable-pip-version-check huggingface_hub \
                   && hf download '$1' --local-dir '/models/$2' \
                   && chown -R $(id -u):$(id -g) '/models/$2'"
    fi
}

download() {
    local repo="$1"
    local name
    name="$(basename "$repo")"
    echo "  $repo -> $MODEL_PATH/$name"
    hf_download "$repo" "$name"
}

download "nvidia/Qwen3.6-35B-A3B-NVFP4"      # ~22 GB — target model
download "z-lab/Qwen3.6-35B-A3B-DFlash"      # ~0.8 GB — speculative draft model

echo ""
echo "Models ready. You can now run: docker compose up -d"
