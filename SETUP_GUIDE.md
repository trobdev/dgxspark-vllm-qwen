# DGX Spark LLM Stack — Setup Guide

End-to-end setup from a fresh DGX Spark to a running Qwen3.6-35B-A3B-NVFP4 inference server
accessible from your network, with Claude Code integration.

**Estimated time:** 1–2 hours (dominated by model download and Docker image build)

---

## Prerequisites

- Ubuntu 22.04 or 24.04
- Internet access from the DGX Spark
- ~50 GB free disk space (30 GB model + Docker image overhead)
- User with `sudo` privileges

---

## Phase 1 — System dependencies

> **On a stock DGX Spark, you can skip almost all of Phase 1.** A factory DGX Spark runs
> DGX OS (an Ubuntu 24.04-based image) that ships with the NVIDIA driver, CUDA toolkit,
> cuDNN, and TensorRT preinstalled, plus **Docker Engine and the NVIDIA Container Toolkit /
> Container Runtime preinstalled and already configured** — NVIDIA describes it as "ready to
> use out of the box." Verify what's present before installing anything:
>
> ```bash
> docker --version            # Docker Engine present
> nvidia-ctk --version        # NVIDIA Container Toolkit present
> nvidia-smi                  # Host driver — confirm 580.x (required for NVFP4)
> docker run --rm --gpus=all \
>   nvcr.io/nvidia/cuda:13.0.1-devel-ubuntu24.04 nvidia-smi   # GPU visible in containers
> ```
>
> If all four succeed, **skip to [Phase 2 — Build the vLLM image](#phase-2--build-the-vllm-image).**
> The install steps below (Steps 1–2) are only needed on a re-imaged machine or a non-DGX-OS
> system where these components are missing.

### Step 1: Install Docker Engine (only if missing)

```bash
sudo apt-get remove -y docker docker-engine docker.io containerd runc

sudo apt-get update
sudo apt-get install -y ca-certificates curl gnupg
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg \
  | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
sudo chmod a+r /etc/apt/keyrings/docker.gpg

echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
  https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo "$VERSION_CODENAME") stable" \
  | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io \
  docker-buildx-plugin docker-compose-plugin

sudo usermod -aG docker $USER
newgrp docker
```

Verify:
```bash
docker --version
```

---

### Step 2: Install NVIDIA Container Toolkit (only if missing)

```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

Verify the GPU is visible in containers:
```bash
docker run --rm --runtime=nvidia --gpus all ubuntu nvidia-smi
```

You should see your GB10 Grace Blackwell listed. If this fails, resolve the NVIDIA
toolkit issue before continuing.

---

## Phase 2 — Build the vLLM image

The DGX Spark requires a custom vLLM build (`eugr/spark-vllm-docker`) due to the GB10's
driver stack. The image must be **vLLM ≥ 0.19** for NVFP4 support on the Blackwell
architecture.

> **No driver modifications required.** Driver 580.x is fully compatible. This is a
> Docker image build only.

```bash
# Clone anywhere — this repo is only needed for the build, not for running the stack
cd ~
git clone https://github.com/eugr/spark-vllm-docker.git
cd spark-vllm-docker
./build-and-copy.sh
```

**Do not run `docker build` directly** — the Dockerfile requires a `build-metadata.yaml`
that only `build-and-copy.sh` generates. The script also auto-downloads prebuilt
FlashInfer and vLLM wheels from the repo's GitHub releases before building the runner
image, which means for the DGX Spark's `12.1a` architecture (GB10 = sm_121) it skips
the full compile entirely and just assembles the final image — typically 15–20 minutes
instead of several hours.

Once complete, the repo directory is no longer needed. The image lives in Docker's store.

Verify the build succeeded and is tagged correctly:
```bash
docker images vllm-node
```

---

## Phase 3 — Project setup

### Step 3: Configure the environment

Create your `.env` from the provided template and add your HuggingFace token:

```bash
cd ~/llm-stack
cp .env.example .env
```

Then edit `.env` and set your values:
```
MODEL_PATH=/data/models
HF_TOKEN=hf_your_token_here
```

Generate a token at https://huggingface.co/settings/tokens, and accept the model
license on the [model's HuggingFace page](https://huggingface.co/nvidia/Qwen3.6-35B-A3B-NVFP4)
first — the repo is gated and the download will 403 otherwise.

> `.env` is gitignored. Never commit your real token.

Ensure the model directory exists:
```bash
sudo mkdir -p /data/models
sudo chown $USER:$USER /data/models
```

---

## Phase 4 — Download model weights

```bash
chmod +x ~/llm-stack/download-models.sh
cd ~/llm-stack
./download-models.sh
```

This downloads `nvidia/Qwen3.6-35B-A3B-NVFP4` (~22 GB). The script skips models
already present, so it is safe to interrupt and re-run if the connection drops.

Verify the download:
```bash
ls /data/models/Qwen3.6-35B-A3B-NVFP4/
```

---

## Phase 5 — Start the stack

```bash
cd ~/llm-stack
docker compose up -d
```

Watch the model load:
```bash
docker compose logs -f vllm-coding
```

A healthy first startup takes ~3–5 minutes (weights load, the KV cache pool is allocated,
and CUDA graphs compile; the healthcheck allows up to 5 minutes) and ends with:
```
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000
```

Verify both services are healthy:
```bash
docker compose ps
```

Test the API:
```bash
curl http://localhost:8001/health

curl http://localhost:8001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "coding",
    "messages": [{"role": "user", "content": "Say hello in one sentence."}],
    "max_tokens": 50
  }'
```

---

## Phase 6 — Claude Code integration

Claude Code can be pointed at the local vLLM server instead of (or alongside) Anthropic's
API. The switch is done entirely via environment variables — no config file editing needed.

### Add shell aliases

In `~/.bashrc` or `~/.zshrc`:

```bash
# Local DGX model — overrides Anthropic API endpoint at Claude Code startup
alias claude-local='ANTHROPIC_BASE_URL=http://<dgx-ip>/coding \
  ANTHROPIC_API_KEY=none \
  ANTHROPIC_DEFAULT_OPUS_MODEL=coding \
  ANTHROPIC_DEFAULT_SONNET_MODEL=coding \
  ANTHROPIC_DEFAULT_HAIKU_MODEL=coding \
  claude'

# Anthropic SaaS — uses your real ANTHROPIC_API_KEY from the environment
alias claude-saas='claude'
```

Replace `<dgx-ip>` with your DGX Spark's IP address (find it with `ip addr show`).

Reload your shell:
```bash
source ~/.bashrc   # or ~/.zshrc
```

### How it works

- `ANTHROPIC_BASE_URL` redirects all Claude API calls to your vLLM server.
  vLLM exposes both an OpenAI-compatible endpoint (`/v1/chat/completions`) and an
  Anthropic-compatible endpoint (`/v1/messages`) on the same port.
- The `ANTHROPIC_*_MODEL` vars map Claude Code's internal model names (opus, sonnet, haiku)
  to the served model name `coding` so vLLM recognizes the request.
- `ANTHROPIC_BASE_URL` is read once at Claude Code startup. To switch endpoints,
  exit and relaunch with the other alias.

### Verify Claude Code can reach the model

```bash
claude-local --version   # should start without auth errors
```

Then ask it something:
```bash
claude-local "Write a Python function that reverses a linked list"
```

---

## Phase 7 — Optional: system autostart

To start the stack automatically after a reboot:

```bash
sudo tee /etc/systemd/system/llm-stack.service > /dev/null <<EOF
[Unit]
Description=DGX Spark LLM Stack
After=docker.service
Requires=docker.service

[Service]
Type=oneshot
RemainAfterExit=yes
WorkingDirectory=/home/$USER/llm-stack
ExecStart=/usr/bin/docker compose up -d
ExecStop=/usr/bin/docker compose down
User=$USER

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable llm-stack
```

---

## Troubleshooting

### Model fails to load (OOM)

If you see a CUDA out-of-memory error, reduce the KV cache allocation:

```bash
# In docker-compose.yml, under vllm-coding command, change:
#   --max-model-len 65536        (down from 131072)
# or:
#   --gpu-memory-utilization 0.80  (down from 0.85)

docker compose up -d --force-recreate vllm-coding
```

### NVFP4 / Blackwell kernel errors on startup

If vLLM fails to load the NVFP4 weights or reports unsupported quantization/kernel errors,
your `vllm-node:latest` image predates NVFP4/Blackwell support. Rebuild it:

```bash
cd ~/spark-vllm-docker
git pull
./build-and-copy.sh
docker compose -f ~/llm-stack/docker-compose.yml up -d --force-recreate vllm-coding
```

### Claude Code returns auth errors with `claude-local`

Confirm the model is healthy first:
```bash
curl http://<dgx-ip>/coding/v1/models
```

If that returns a model list, the issue is likely the `ANTHROPIC_BASE_URL` path.
vLLM's Anthropic endpoint is at the root of the base URL — the path should be
`http://<dgx-ip>/coding` (not `/coding/v1`).

### nvidia-smi shows no GPU in container

```bash
docker info | grep -i runtime
# Should show: Runtimes: nvidia runc
# If not:
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

### Check GPU memory during inference

```bash
nvidia-smi --query-gpu=memory.used,memory.free,memory.total --format=csv
```

With the model loaded you should see ~100–110 GB used out of 128 GB.
