# DGX Spark LLM Stack — Setup Guide

This guide takes a fresh DGX Spark to a running Qwen3.6-35B-A3B-NVFP4 inference server
that your network can reach, with Claude Code connected. It follows the same steps as the
[README quick start](README.md#quick-start), adding a verification command at each step
and a troubleshooting section.

**Estimated time:** 1–1.5 hours, mostly the vLLM image build (~40 min) and the model download.

---

## Prerequisites

- **Hardware:** NVIDIA GB10 with 128 GB unified memory (DGX Spark or an OEM equivalent).
- **OS:** DGX OS, which is Ubuntu 24.04-based, or Ubuntu 24.04 with the NVIDIA 580.x driver.
- **Network:** internet access from the Spark.
- **Disk:** ~50 GB free (~23 GB of models plus a ~25 GB image).
- **Access:** a user with `sudo`.

---

## Phase 1 — System dependencies

> **On a stock DGX Spark you can skip almost all of Phase 1.** DGX OS ships with the NVIDIA
> driver, CUDA, Docker Engine, and the NVIDIA Container Toolkit, all preinstalled and
> configured. Check before installing anything:
>
> ```bash
> docker --version            # Docker Engine present
> docker compose version      # Compose v2 present
> nvidia-ctk --version        # NVIDIA Container Toolkit present
> nvidia-smi                  # Driver: must be 580.x (required for NVFP4)
> docker run --rm --gpus=all \
>   nvcr.io/nvidia/cuda:13.0.1-devel-ubuntu24.04 nvidia-smi   # GPU visible in containers
> ```
>
> If all five succeed, **skip to [Phase 2](#phase-2--build-the-vllm-image).** Steps 1–2
> below are only for a re-imaged machine or a non-DGX-OS install.
>
> **Do not update the GPU driver.** On a DGX Spark the driver is part of the platform image.

### Step 1: Install Docker Engine (only if missing)

```bash
sudo apt-get update
sudo apt-get install -y ca-certificates curl
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc

echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] \
  https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo "$VERSION_CODENAME") stable" \
  | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io \
  docker-buildx-plugin docker-compose-plugin

sudo usermod -aG docker $USER     # then log out and back in
```

### Step 2: Install the NVIDIA Container Toolkit (only if missing)

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

---

## Phase 2 — Build the vLLM image

The GB10 (`sm_121`) needs a vLLM build with Blackwell and NVFP4 kernels. Build one with
[`eugr/spark-vllm-docker`](https://github.com/eugr/spark-vllm-docker). The image runs on
the stock 580.x driver, so no driver changes are needed.

**This compose file needs vLLM 0.26 or newer** (DFlash speculative decoding and
`--safetensors-load-strategy`). It was validated on **v0.30.0**, and the command below
reproduces that build:

```bash
git clone https://github.com/eugr/spark-vllm-docker.git ~/spark-vllm-docker
cd ~/spark-vllm-docker
git checkout f7d6e3b       # build-script version used for the validated image
./build-and-copy.sh -t vllm-node-v2 --vllm-ref v0.30.0 --apply-preset-vllm-prs -j 8
```

- **Time:** it compiles vLLM from source, which measured **39m42s**. Don't interrupt it,
  because a partial build leaves a broken image.
- **`-t vllm-node-v2` is required.** The script's default tag is `vllm-node`, but
  `docker-compose.yml` runs `vllm-node-v2:latest`. Without the flag the build succeeds and
  then the stack can't find the image.
- **Use the script, not `docker build`.** The Dockerfile needs a `build-metadata.yaml` that
  only `build-and-copy.sh` generates.
- **FlashInfer comes from the build repo's prebuilt wheel**, which was 0.7.1 at validation
  time. Add `--flashinfer-ref v0.7.1` to pin it too, at the cost of a FlashInfer source build.
- **Unpinned alternative:** `./build-and-copy.sh -t vllm-node-v2` downloads the build
  repo's current prebuilt vLLM and FlashInfer wheels instead of compiling. It takes ~15–20
  minutes but gives a version not validated here, so run `bench/validate.py` after Phase 5.
- **Memory:** a source build shares unified memory with everything else. Next to a resident
  model (~91 GiB) an `-j 8` build triggered the OOM killer, so keep vLLM stopped while building.

Verify:
```bash
docker run --rm --runtime=nvidia vllm-node-v2:latest \
  python3 -c "import vllm, flashinfer; print(vllm.__version__, flashinfer.__version__)"
# validated build prints: 0.30.1.dev1+gbf1979fbd... 0.7.1
```

Once the image is built, `~/spark-vllm-docker` is only needed for future rebuilds.

---

## Phase 3 — Project setup

```bash
git clone https://github.com/trobdev/dgxspark-vllm-qwen.git ~/dgxspark-vllm-qwen
cd ~/dgxspark-vllm-qwen
cp .env.example .env
sed -i "s/^LLM_API_KEY=.*/LLM_API_KEY=$(openssl rand -hex 32)/" .env
```

Check `.env`:
```bash
cat .env | grep -v '^#'
# LLM_API_KEY=<64 hex characters>   required; clients send this
# MODEL_PATH=/data/models           where weights are stored
# HF_TOKEN=                         optional; both models are public
```

> `.env` is gitignored. Never commit it.

Create the model directory:
```bash
sudo mkdir -p /data/models
sudo chown $USER:$USER /data/models
```

---

## Phase 4 — Download model weights

```bash
cd ~/dgxspark-vllm-qwen
./download-models.sh
```

This downloads two public (Apache-2.0) models to `MODEL_PATH`:

| Model | Size | Role |
|---|---|---|
| `nvidia/Qwen3.6-35B-A3B-NVFP4` | ~22 GB | the served model |
| `z-lab/Qwen3.6-35B-A3B-DFlash` | ~0.8 GB | speculative-decoding draft model; required by the shipped config |

The script reads `.env`. It uses the `hf` CLI if it's installed; otherwise it runs it in a
throwaway `python:3.12-slim` container, because DGX OS's system Python refuses
`pip install`. If the download is interrupted, re-run the script and it resumes.

Verify:
```bash
ls /data/models/
# Qwen3.6-35B-A3B-DFlash  Qwen3.6-35B-A3B-NVFP4
du -sh /data/models/*
```

---

## Phase 5 — Start the stack

```bash
cd ~/dgxspark-vllm-qwen
docker compose up -d
docker compose logs -f vllm-coding
```

A first start takes **341–461s**. It loads weights, allocates the KV cache pool, compiles
CUDA graphs, and runs FlashInfer autotuning. Later restarts reuse the `vllm-cache` volume
and take 285–294s. The healthcheck allows 600s before it counts failures. Startup ends with:

```
INFO:     Application startup complete.
```

Check that all three services are healthy. nginx and the shim start only after vLLM is healthy:

```bash
docker compose ps
# vllm-coding, anthropic-shim and nginx should all show (healthy)
```

Then test the API directly on the host, then through nginx with your key:

```bash
curl http://localhost:8001/health      # direct vLLM, loopback only, no key

KEY=$(grep ^LLM_API_KEY= .env | cut -d= -f2)
curl http://localhost/coding/v1/chat/completions \
  -H "Authorization: Bearer $KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "coding",
    "messages": [{"role": "user", "content": "Say hello in one sentence."}],
    "max_tokens": 50,
    "chat_template_kwargs": {"enable_thinking": false}
  }'
```

`enable_thinking: false` matters for a quick test. This is a reasoning model, so with a
small `max_tokens` and thinking on, the whole budget goes to reasoning and `content`
comes back empty.

Finally, run the full validation suite:

```bash
python3 bench/validate.py     # expect 7/7 passed
```

---

## Phase 6 — Connect Claude Code and other clients

See [README → Connecting clients](README.md#connecting-clients). The short version for Claude Code:

```bash
alias claude-local='env -u ANTHROPIC_API_KEY \
  ANTHROPIC_BASE_URL=http://<dgx-ip>/coding \
  ANTHROPIC_AUTH_TOKEN=<LLM_API_KEY> \
  ANTHROPIC_DEFAULT_OPUS_MODEL=coding \
  ANTHROPIC_DEFAULT_SONNET_MODEL=coding \
  ANTHROPIC_DEFAULT_HAIKU_MODEL=coding \
  claude'
```

`<dgx-ip>` is the Spark's address (`hostname -I`), or `localhost` if Claude Code runs on the
Spark. Claude Code requests `/coding/v1/messages`. nginx checks the key and routes the
request to the shim, which moves Claude Code's `system`-role messages into the top-level
`system` field before forwarding to vLLM.

**Don't point Claude Code at port 8001 directly.** That skips the shim, and vLLM returns 400
for the `system`-role messages.

Verify with a real request:
```bash
claude-local -p "Reply with exactly: LOCAL-OK"
```

---

## Phase 7 — Optional: system autostart

Run this from the repository directory, so that `$PWD` expands to the right path:

```bash
cd ~/dgxspark-vllm-qwen
sudo tee /etc/systemd/system/llm-stack.service > /dev/null <<EOF
[Unit]
Description=DGX Spark LLM Stack
After=docker.service network-online.target
Requires=docker.service

[Service]
Type=oneshot
RemainAfterExit=yes
WorkingDirectory=$PWD
ExecStart=/usr/bin/docker compose up -d
ExecStop=/usr/bin/docker compose down
User=$USER

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable --now llm-stack
```

---

## Troubleshooting

### `docker compose up` says `required variable LLM_API_KEY is missing a value`

The key in `.env` is empty. Generate one:
```bash
sed -i "s/^LLM_API_KEY=.*/LLM_API_KEY=$(openssl rand -hex 32)/" .env
```

### Requests return `401 missing or wrong API key`

The client isn't sending `LLM_API_KEY`. Send it as `Authorization: Bearer <key>` (OpenAI
clients, `ANTHROPIC_AUTH_TOKEN`) or as `x-api-key: <key>`. If you changed the key, restart
nginx with `docker compose up -d --force-recreate nginx`, because the key is read at startup.

### vLLM fails with an unrecognised argument

For example, `unrecognized arguments: --safetensors-load-strategy`, or an unknown
speculative method `dflash`. Your image is older than vLLM 0.26. Rebuild it as in Phase 2.

### Model fails to load (out of memory)

```bash
# In docker-compose.yml, under vllm-coding command, change:
#   --gpu-memory-utilization 0.70  (down from 0.75)
# or, only if that is not enough:
#   --max-model-len 131072         (down from 262144)
#
# Prefer lowering utilization first: 262144 is set deliberately above the 200K window
# Claude Code assumes, so its auto-compaction fires before the server's limit.

docker compose up -d --force-recreate vllm-coding
```

Also check what else is using memory with `free -h`. Everything on the box shares the same 128 GB pool.

### NVFP4 / Blackwell kernel errors on startup

The image predates NVFP4 or Blackwell support. Rebuild as in Phase 2, keeping a rollback first:

```bash
docker tag vllm-node-v2:latest vllm-node:v2-previous
```

### Claude Code can't connect

1. Confirm the model is reachable through nginx:
   ```bash
   curl -H "Authorization: Bearer <LLM_API_KEY>" http://<dgx-ip>/coding/v1/models
   ```
2. Check the base URL. It must be `http://<dgx-ip>/coding`, not `/coding/v1`, because
   Claude Code appends `/v1/messages` itself.
3. Check that you used `ANTHROPIC_AUTH_TOKEN`, not `ANTHROPIC_API_KEY`.

### Port 80 not reachable from another machine

Check `docker compose ps` shows nginx publishing `0.0.0.0:80->80/tcp`. Then check that no
host firewall or network ACL between the machines blocks port 80.

### nvidia-smi shows no GPU in container

```bash
docker info | grep -i runtime
# Should list nvidia. If not:
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

### Checking GPU memory

On GB10, `nvidia-smi --query-gpu=memory.used` reports `[N/A]` because memory is unified.
Use the per-process view:

```bash
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
# VLLM::EngineCore should show roughly 90,000–93,000 MiB
free -h     # whole-box view
```
