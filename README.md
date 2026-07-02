# DGX Spark — LLM Inference Stack

Single-model vLLM inference server behind Nginx, managed with Docker Compose.
Turns a DGX Spark into a private, OpenAI- and Anthropic-compatible coding API for
Claude Code and IDE assistants. vLLM serves both the OpenAI API (`/v1/chat/completions`)
and the Anthropic API (`/v1/messages`) natively, but Claude Code injects a
`system`-role message inside the `messages` array — something vLLM rejects with a 400.
A dependency-free, stdlib-only normalizing shim hoists those system messages into
vLLM's top-level `system` field before forwarding, preserving the Anthropic response
shape (including `thinking` blocks) so no separate translation proxy is needed.
OpenAI-compatible clients connect directly through Nginx — they bypass the shim entirely
because vLLM speaks OpenAI natively.

**Model:** `nvidia/Qwen3.6-35B-A3B-NVFP4` — 35B MoE (3B active), Blackwell NVFP4, 128K context, tool-calling.
The stack is model-agnostic; see [Swapping the model](#swapping-the-model) to use a different one.

## Documentation

| Doc | Purpose |
|-----|---------|
| **README.md** (this file) | Quick start, API access, day-to-day operations |
| [SETUP_GUIDE.md](SETUP_GUIDE.md) | End-to-end setup from a fresh DGX Spark (Docker, NVIDIA toolkit, vLLM image build) |
| [IMPLEMENTATION_GUIDE.md](IMPLEMENTATION_GUIDE.md) | The "why" — architecture, design decisions, and the reasoning behind each config choice |

## Prerequisites

- Docker Engine with the NVIDIA Container Toolkit installed and the runtime configured
- `vllm-node:latest` image built locally (see SETUP_GUIDE.md — must be vLLM ≥ 0.19 for NVFP4)
- ~30 GB of free disk space for model weights

> On a stock **DGX Spark**, Docker Engine and the NVIDIA Container Toolkit/runtime are
> preinstalled and configured out of the box — the first bullet is already done. See
> [SETUP_GUIDE.md → Phase 1](SETUP_GUIDE.md#phase-1--system-dependencies) for the quick
> verification commands.

---

## First-time setup

### 1. Configure your environment

```bash
cp .env.example .env
# Edit .env and set HF_TOKEN to your HuggingFace token
```

The model is a gated HuggingFace repo — accept the license on the
[model page](https://huggingface.co/nvidia/Qwen3.6-35B-A3B-NVFP4) before downloading.

### 2. Download model weights

```bash
chmod +x download-models.sh
./download-models.sh
```

### 3. Start the stack

```bash
docker compose up -d
```

Watch startup (first load takes ~3–5 minutes — weights load, the KV cache pool is
allocated, and CUDA graphs compile; the healthcheck allows up to 5 minutes):

```bash
docker compose logs -f vllm-coding
```

A healthy startup ends with:
```
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000
```

---

## API access

The model is served at an OpenAI-compatible endpoint and an Anthropic-compatible endpoint:

| Protocol   | URL                                       | Model name |
|------------|-------------------------------------------|------------|
| OpenAI     | `http://<dgx-ip>/coding/v1`               | `coding`   |
| Anthropic  | `http://<dgx-ip>/coding` (base URL only)  | `coding`   |

### OpenAI SDK

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://192.168.1.100/coding/v1",
    api_key="none",
)

response = client.chat.completions.create(
    model="coding",
    messages=[{"role": "user", "content": "Write a binary search in Python."}],
)
print(response.choices[0].message.content)
```

### With thinking mode enabled (extended reasoning)

```python
response = client.chat.completions.create(
    model="coding",
    messages=[{"role": "user", "content": "Design a thread-safe LRU cache in Python."}],
    extra_body={"chat_template_kwargs": {"enable_thinking": True}},
)
```

---

## Claude Code integration

Claude Code can use this server as a drop-in replacement for Anthropic's API, or you can
switch between local and SaaS per session using shell aliases.

### Shell aliases (recommended)

Add to `~/.bashrc` or `~/.zshrc`:

```bash
# Use local DGX model
alias claude-local='ANTHROPIC_BASE_URL=http://<dgx-ip>/coding \
  ANTHROPIC_API_KEY=none \
  ANTHROPIC_DEFAULT_OPUS_MODEL=coding \
  ANTHROPIC_DEFAULT_SONNET_MODEL=coding \
  ANTHROPIC_DEFAULT_HAIKU_MODEL=coding \
  claude'

# Use Anthropic SaaS models (Opus, Sonnet, etc.) — uses real API key from env
alias claude-saas='claude'
```

Replace `<dgx-ip>` with your DGX Spark's IP address.

**Notes:**
- `ANTHROPIC_BASE_URL` is read once at Claude Code startup — restart Claude Code to switch.
- The `ANTHROPIC_*_MODEL` vars map Claude Code's internal model references to your served model name.
- vLLM must be running and healthy before starting Claude Code against it.

### IDE / Cursor / Continue.dev

Point at the OpenAI-compatible endpoint:

| Tool         | Setting          | Value                              |
|--------------|------------------|------------------------------------|
| Cursor       | OpenAI Base URL  | `http://<dgx-ip>/coding/v1`        |
| Continue.dev | `apiBase`        | `http://<dgx-ip>/coding/v1`        |

Model name: `coding`, API key: `none`.

> **Continue.dev:** Also add `systemMessage: "/no_think"` to your model entry in `~/.continue/config.yaml`. The model reasons by default and Continue's OpenAI adapter cannot handle the `reasoning_content` delta field vLLM emits for thinking tokens — the directive suppresses it.

---

## Day-to-day operations

| Task                    | Command                                       |
|-------------------------|-----------------------------------------------|
| Start                   | `docker compose up -d`                        |
| Stop                    | `docker compose down`                         |
| Restart model only      | `docker compose restart vllm-coding`          |
| View live logs          | `docker compose logs -f vllm-coding`          |
| Check GPU usage         | `nvidia-smi`                                  |
| Health check            | `curl http://localhost:8001/health`           |
| Update vLLM image       | `cd ~/spark-vllm-docker && git pull && ./build-and-copy.sh` |

---

## GPU memory

With `--gpu-memory-utilization 0.85` vLLM commits ~108 GB of the 128 GB pool,
leaving the rest as headroom:

| Allocation        | Approx size  |
|-------------------|--------------|
| NVFP4 weights     | ~22 GB       |
| KV cache (128K)   | ~87 GB       |
| Free headroom     | ~19 GB       |

If the model OOMs on startup, reduce `--max-model-len` to `65536` or lower
`--gpu-memory-utilization` to `0.80` in `docker-compose.yml`.

---

## Security

This stack ships with **no authentication and no TLS** by default — it is built for a
single developer or a trusted LAN.

- **No API auth.** Nginx forwards every `/coding/` request and the served key is `none`.
  Anyone who can reach the port has full use of the model, and your prompts (including
  source code) pass through it. Add a bearer-token check to the `/coding/` block in
  [nginx/nginx.conf](nginx/nginx.conf) before exposing it beyond a network you control —
  see [IMPLEMENTATION_GUIDE.md → Configure Nginx](IMPLEMENTATION_GUIDE.md#step-5--configure-nginx).
- **No TLS.** Port 80 is plaintext. Enable the HTTPS block and add certs if the segment
  isn't fully trusted.
- **Debug port is loopback-only.** vLLM's direct port is published as `127.0.0.1:8001`
  so it can't be reached from the network (it would bypass Nginx). Keep it that way.
- **Do not port-forward `80`/`443`/`8001` to the internet** without adding auth and TLS first.

## Swapping the model

1. Download new weights to `MODEL_PATH`
2. Update `--model` and `--served-model-name` in `docker-compose.yml`
3. `docker compose up -d --force-recreate vllm-coding`

---

## Important: vLLM image requirement

NVFP4 quantization on the GB10 Blackwell architecture requires **vLLM ≥ 0.19**.
The `vllm-node:latest` image must be rebuilt from `eugr/spark-vllm-docker` if your existing
build predates NVFP4 support. See [SETUP_GUIDE.md](SETUP_GUIDE.md) for rebuild instructions.
