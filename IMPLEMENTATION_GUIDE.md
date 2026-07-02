# DGX Spark vLLM Stack — Implementation Guide

## Why This Stack Exists

Running large language models locally on developer hardware has historically meant accepting a painful tradeoff: models small enough to fit in GPU VRAM are too weak for serious coding work, while models capable enough require multi-GPU server racks. This stack exists because the DGX Spark changes that equation — and takes some deliberate effort to unlock properly.

The architecture here has one goal: make a 35-billion-parameter reasoning model feel like a cloud API, fast enough for interactive use, running entirely on a single machine you own.

---

## Architecture Overview

```
┌──────────────────────────────────────────────────────────────┐
│                        CLIENTS                               │
│  Claude Code / IDEs              curl / Continue / OpenAI    │
│  (claude-local alias)            (compatible apps)           │
└───────────────┬──────────────────────────┬───────────────────┘
   :80 /coding (Anthropic)      :80 /coding/v1 (OpenAI)
                └──────────────┬───────────┘
                               ▼
             ┌─────────────────────────────────────┐
             │         Nginx  :80 / :443           │
             │  /coding/v1/messages → shim:8080    │
             │  /coding/other → vllm-coding:8000   │
             │  (reverse proxy, streaming)         │
             └───────────────────┬─────────────────┘
                                 │
             ┌─────────────────▼──────────────────┐
             │  docker network: llm-net            │
             │  ┌──────────┐  ┌────────────────┐  │
             │  │  shim   │  │   vLLM — coding│  │
             │  │ :8080   │  │   :8001        │  │
             │  │ hoists   │  │ Qwen3.6-35B    │  │
             │  │ system[] │  │ NVFP4 · 128K   │  │
             │  └────┬─────┘  │ ctx · MoE     │  │
             │       │        └────────────────┘  │
             │       └──────────────┬─────────────┘
             │                      │
             │  shared vol: /data/models
             └───────────────────┬─────────────────┘
                                 │
             ┌───────────────────▼─────────────────┐
             │   NVIDIA Container Runtime           │
             │   CUDA · NVFP4 · PagedAttention      │
             └───────────────────┬─────────────────┘
                                 │
             ┌───────────────────▼─────────────────┐
             │   DGX Spark — GB10 Grace Blackwell   │
             │   128 GB unified · 273 GB/s          │
             └─────────────────────────────────────┘
```

### GPU Memory Budget (128 GB, gpu_util = 0.85 → ~108 GB committed)

```
[████████████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░]
 17%  ~22 GB       68%  ~87 GB KV cache              15% free
 NVFP4 weights     (PagedAttention pool)              ~19 GB
```

---

## The Platform: Why the DGX Spark Changes the Rules

Traditional GPU-based inference has a hard constraint: model weights must fit in VRAM, and VRAM is a small, expensive, physically separate pool — typically 24–80 GB on a workstation. On those machines, a 35B parameter model in full precision (BF16) requires ~70 GB and simply doesn't fit. Even with aggressive quantization you're pushing limits, and a KV cache large enough for 128K context is out of the question.

The DGX Spark's GB10 Grace Blackwell changes this with unified memory. The CPU and GPU do not have separate memory pools — there is a single 128 GB pool addressed by both. This is not a software trick like CPU offloading (which shuttles weights across PCIe); it is the hardware architecture, and the GPU's tensor cores address the entire 128 GB directly.

The advantage here is **capacity, not raw bandwidth.** The unified pool runs at ~273 GB/s (LPDDR5X) — modest next to a discrete GPU's VRAM (an H100 moves >3 TB/s). What the Spark buys you is the ability to hold a 35B model *and* a large KV cache in one pool at all, on a single device, without the multi-GPU rack that capacity would otherwise demand. You trade peak throughput for the fact that the workload fits — which is why generation lands at ~100–110 tokens/second (see [Throughput Expectations](#throughput-expectations-and-limitations)) rather than cloud-cluster speeds.

What this means in practice:
- A 35B model fits with room to spare for a massive KV cache
- The KV cache can grow to 87 GB — enough to hold hundreds of concurrent long-context conversations
- There is no PCIe copy between CPU and GPU memory — both address the same physical pool

The GB10 supports NVFP4 quantization, which is what the `Qwen3.6-35B-A3B-NVFP4` checkpoint uses. The primary benefit is **memory**: weights stored in 4-bit take ~22 GB instead of ~70 GB at BF16 — this is what makes the full 35B model fit alongside the large KV cache. On the compute side, the current vLLM build runs those weights via the **Marlin weight-only path**: weights stay in 4-bit in memory (the size advantage is fully preserved), but at inference time they are dequantized to BF16 before the matrix multiply, which runs on standard BF16 tensor cores. For interactive single-developer decode on a sparse MoE (only ~3B parameters active per token), this distinction barely matters: the bottleneck is reading the small set of active weights from memory, not the speed of the multiply. Measured throughput with this setup is 100–110 tokens/second — fast enough that a 500-token response streams in under 6 seconds.

> **Risk — Driver Version:** NVFP4 requires driver 580.x or later to load the quantized checkpoint. Verify with `nvidia-smi` before investing time in anything else.

---

## The Model: Qwen3.6-35B and the MoE Tradeoff

The model choice — `Qwen3.6-35B-A3B-NVFP4` — reflects a specific engineering tradeoff between capability and inference cost that is worth understanding explicitly.

Mixture of Experts (MoE) is the key architectural insight. A standard dense model with 35B parameters activates all 35B on every token it generates. A MoE model partitions its parameters into "experts" (specialized sub-networks) and routes each token through only a small subset of them. Qwen3.6-35B has 35B total parameters, but only ~3B are active per forward pass — the "A3B" in the model name (its config routes each token through 8 of 256 experts). This means:

- Memory footprint of a 35B model (~22 GB at NVFP4)
- Compute cost of a ~3B model per generated token
- Quality and knowledge depth of a 35B model (because the full 35B of learned knowledge is available, just not all at once)

For a coding task, this is close to ideal. Coding requires a large knowledge base (language syntax, APIs, libraries, idioms across dozens of languages) but any single token generation draws from a small, specialized slice of that knowledge.

**Why not a larger model?** A 120B dense model would require ~120 GB at NVFP4, consuming the entire memory budget with nothing left for KV cache — meaning context would be severely limited, and throughput would collapse. The MoE architecture is what makes 35B quality with practical throughput possible on this hardware.

**Why not a smaller model?** 7–8B models are fast and cheap, but they lack the multi-hop reasoning depth needed for the kind of tasks Claude Code handles: understanding how a change in one file ripples through a codebase, generating correct multi-file diffs, reasoning through complex dependency trees. The capability gap between 8B and 35B on hard coding tasks is significant.

NVFP4 quantization is applied on top. NVIDIA released this checkpoint with their own quantization tooling, including calibration data that ensures the 4-bit representation accurately captures the important numerical ranges in the original weights. The model's outputs are not meaningfully degraded relative to the FP16 original for coding tasks.

### Using a Different Model

The stack is not locked to Qwen3.6-35B. Any model with an OpenAI-compatible vLLM endpoint will work — the Nginx reverse proxy is model-agnostic. What does need to change if you swap models:

- **`docker-compose.yml`**: update `--model` to point at the new model directory, `--served-model-name` if you want a different alias, and remove `--reasoning-parser` / `--tool-call-parser` if the new model doesn't support Qwen3's reasoning/tool-call format.
- **`moe-configs/`**: the pre-tuned kernel config is GB10-specific and MoE-specific. If the new model is a dense transformer (e.g. Llama, Mistral), delete or empty this directory — vLLM will ignore it or find no matching config.
- **Memory budget**: recalculate `--gpu-memory-utilization` based on the new model's weight footprint. A model that uses more of the 128 GB for weights leaves less for KV cache, which reduces achievable context length and concurrency.
- **Quantization**: NVFP4 is specific to NVIDIA-released checkpoints with calibration data baked in. Most open-source models are available in GPTQ, AWQ, or GGUF. vLLM supports GPTQ and AWQ natively; pick the format that the vLLM version you built supports.

The `--reasoning-parser`, `--tool-call-parser`, and `--language-model-only` flags in the compose file are Qwen3-specific and should be removed or adjusted for other models.

---

## Prerequisites

| Requirement | Detail |
|---|---|
| Hardware | DGX Spark (GB10 Grace Blackwell, 128 GB unified memory) |
| NVIDIA Driver | 580.x — required for NVFP4 support |
| Docker | 24+ with Compose v2 |
| NVIDIA Container Toolkit | nvidia-container-toolkit installed |
| vLLM Image | ≥ 0.19 — required for NVFP4 (built locally via `eugr/spark-vllm-docker`) |
| Model | Any vLLM-compatible model; this guide uses `nvidia/Qwen3.6-35B-A3B-NVFP4` (gated HF repo — needs HF token) |
| Disk | ~25–30 GB free under `/data/models` for Qwen3.6-35B-A3B-NVFP4; more for larger models |

---

## Step 1 — Build the vLLM Docker Image

vLLM is the inference engine at the core of this stack. It is not simply a model-serving framework — it contains a purpose-built memory manager (PagedAttention), speculative decoding primitives, and hardware-specific kernel implementations that make the difference between a model that runs and a model that runs at production-grade throughput.

The official vLLM Docker image does not include kernels built for the GB10 Blackwell architecture, and does not yet support NVFP4 inference. The `eugr/spark-vllm-docker` build targets Blackwell and produces a vLLM version ≥0.19, which is the first with NVFP4 support. By default `build-and-copy.sh` downloads prebuilt vLLM and FlashInfer wheels from the repo's GitHub releases rather than compiling from source, so for the DGX Spark's `sm_121` architecture it assembles the final image in roughly 15–20 minutes instead of several hours.

```bash
git clone https://github.com/eugr/spark-vllm-docker
cd spark-vllm-docker
./build-and-copy.sh
```

> **Risk — Build Time:** With prebuilt wheels the build typically takes ~15–20 minutes. If you force a from-source build (e.g. `--rebuild-vllm` or a custom `--vllm-ref`), it compiles CUDA kernels for Blackwell and can take 30–60 minutes — do not interrupt that, as a partial kernel compilation leaves a broken image.

> **Risk — Image Tag:** The `docker-compose.yml` hardcodes `vllm-node:latest`. If you tag the image differently during the build, update the compose file before proceeding.

Verify the build succeeded and vLLM reports the correct version:

```bash
docker run --rm --runtime=nvidia vllm-node:latest python3 -c "import vllm; print(vllm.__version__)"
# Must print 0.19 or higher
```

---

## Step 2 — Download the Model

The repo includes `download-models.sh`, which handles the download idempotently — if the model directory already exists it skips it, so it's safe to re-run after an interrupted download.

```bash
export HF_TOKEN=hf_your_token_here
export MODEL_PATH=/data/models   # must match MODEL_PATH in .env
bash download-models.sh
```

The script installs `huggingface_hub` if needed and downloads `nvidia/Qwen3.6-35B-A3B-NVFP4` to `$MODEL_PATH/Qwen3.6-35B-A3B-NVFP4`.

> **Risk — Disk Space:** The NVFP4 quantized model is ~22 GB. Ensure `/data/models` has 30+ GB free to allow for partial download recovery.

> **Risk — Gated Model:** `nvidia/Qwen3.6-35B-A3B-NVFP4` is a gated HuggingFace repository. You must accept the license terms on the HuggingFace model page before your token will be authorized to download it.

### A Note on FP8 KV Cache — and Why This Stack Doesn't Use It

This is worth explaining explicitly because it's a tempting optimization that would appear to free up significant memory, and the reason to avoid it is non-obvious.

The KV cache stores the computed key/value attention tensors for every token in every active context window. At BF16 (the default), this cache consumes ~87 GB in this stack. FP8 quantization of the KV cache would roughly halve that, freeing ~44 GB — a huge gain.

However, FP8 KV cache quantization requires per-tensor calibration scales (`k_scale`, `v_scale`, `q_scale`) that tell vLLM how to rescale the values before quantizing them. These scales must be computed on a representative dataset and baked into the checkpoint. **The `Qwen3.6-35B-A3B-NVFP4` checkpoint does not include these scales.**

When scales are missing, vLLM falls back to `scale=1.0` — meaning no rescaling. FP8 E4M3 can only represent values up to 448. Any attention tensor value above 448 is silently clipped. In practice this means the model computes attention over subtly wrong values on long contexts: the kind of errors that produce wrong variable names, hallucinated API signatures that look plausible, or off-by-one logic that's hard to catch in review. This is exactly the wrong failure mode for a coding assistant.

The decision is: leave KV cache at BF16, accept the full ~87 GB cost, and preserve accuracy. **Do not add `--kv-cache-dtype fp8` to the vLLM command.**

---

## Step 3 — Set Up Directory Structure

```
llm-stack/
├── docker-compose.yml
├── .env                    ← HF_TOKEN and MODEL_PATH
├── moe-configs/            ← Pre-tuned Triton kernel configs for GB10 (mounted read-only)
├── nginx/
│   ├── nginx.conf          ← Reverse proxy config
│   └── certs/              ← TLS certs (optional)
└── shim/
    └── shim.py             ← Anthropic → vLLM normalizing shim (stdlib-only, dependency-free)
```

```bash
mkdir -p llm-stack/{moe-configs,nginx/certs,shim}
cd llm-stack
echo "HF_TOKEN=hf_your_token_here" > .env
echo "MODEL_PATH=/data/models" >> .env
```

### `docker-compose.yml`

This is the central file that wires everything together. Create it at the root of `llm-stack/`:

```yaml
name: dgx-llm-stack

services:

  # ── Qwen3.6-35B-A3B-NVFP4 coding model ───────
  vllm-coding:
    image: vllm-node:latest  # Built locally via eugr/spark-vllm-docker for DGX Spark — must be vLLM ≥0.19 for NVFP4
    container_name: vllm-coding
    runtime: nvidia
    restart: "no"
    networks:
      - llm-net
    volumes:
      - models:/models
      - huggingface-cache:/root/.cache/huggingface
      - ./moe-configs:/moe-configs:ro
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
      - HF_TOKEN=${HF_TOKEN:-}
      - VLLM_LOGGING_LEVEL=WARNING
      - VLLM_TUNED_CONFIG_FOLDER=/moe-configs
    command:
      - "python3"
      - "-m"
      - "vllm.entrypoints.openai.api_server"
      - "--model"
      - "/models/Qwen3.6-35B-A3B-NVFP4"
      - "--served-model-name"
      - "coding"
      - "--host"
      - "0.0.0.0"
      - "--port"
      - "8000"
      - "--max-model-len"
      - "131072"
      - "--gpu-memory-utilization"
      - "0.85"
      - "--tensor-parallel-size"
      - "1"
      - "--enable-prefix-caching"
      - "--max-num-seqs"
      - "32"
      - "--reasoning-parser"
      - "qwen3"
      - "--enable-auto-tool-choice"
      - "--tool-call-parser"
      - "qwen3_coder"
      - "--language-model-only"    # Skip vision encoder — not needed for coding, frees VRAM
      # NOTE: ngram speculative decoding (--spec-method ngram --spec-tokens 5) was
      # removed — on vLLM 0.22.1rc1 it crashed EngineCore with a KV-cache block
      # accounting assertion (num_required_blocks off by one) on large contexts.
      # Re-enable once on a vLLM release where that's fixed.
    ports:
      - "127.0.0.1:8001:8000"   # debug port — loopback only, never network-reachable
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              capabilities: [gpu]
    healthcheck:
      test: ["CMD", "curl", "-sf", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 5
      start_period: 300s

  # ── Anthropic normalizing shim ─────────────────
  # Claude Code injects a system-role message into messages[]; vLLM's native
  # /v1/messages rejects non-user/assistant roles. This dependency-free stdlib
  # shim hoists system-role messages into the top-level `system` field and
  # forwards to vLLM, keeping the native Anthropic path (preserves `thinking`).
  anthropic-shim:
    image: python:3.12-slim
    container_name: anthropic-shim
    restart: "no"
    networks:
      - llm-net
    volumes:
      - ./shim/shim.py:/app/shim.py:ro
    environment:
      - VLLM_UPSTREAM=http://vllm-coding:8000
    command: ["python3", "/app/shim.py"]
    depends_on:
      vllm-coding:
        condition: service_healthy
    healthcheck:
      test: ["CMD", "python3", "-c", "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8080/healthz').status==200 else 1)"]
      interval: 30s
      timeout: 5s
      retries: 3
      start_period: 10s

  # ── Nginx reverse proxy ────────────────────────
  nginx:
    image: nginx:alpine
    container_name: nginx
    restart: "no"
    networks:
      - llm-net
    volumes:
      - ./nginx/nginx.conf:/etc/nginx/nginx.conf:ro
      - ./nginx/certs:/etc/nginx/certs:ro       # Optional: add TLS certs here
    ports:
      - "80:80"
      - "443:443"
    depends_on:
      vllm-coding:
        condition: service_healthy
      anthropic-shim:
        condition: service_healthy
    healthcheck:
      test: ["CMD", "nginx", "-t"]
      interval: 60s
      timeout: 5s
      retries: 3

networks:
  llm-net:
    driver: bridge

volumes:
  models:
    driver: local
    driver_opts:
      type: none
      o: bind
      device: ${MODEL_PATH:-/data/models}

  huggingface-cache:
    driver: local
```

A few things worth calling out:

**`--language-model-only`** skips loading the vision encoder that ships in the Qwen3 checkpoint. The DGX Spark has no camera; loading the vision encoder wastes ~2 GB of unified memory and adds startup time.

**`--reasoning-parser qwen3` and `--tool-call-parser qwen3_coder`** enable structured output parsing. Qwen3 uses a chain-of-thought reasoning format (`<think>...</think>`) and a specific tool call format; without these parsers vLLM would return the raw tokens and Claude Code would see malformed responses.

**`VLLM_TUNED_CONFIG_FOLDER=/moe-configs`** points vLLM at the mounted `moe-configs/` directory so it can load a pre-tuned Triton config for its fused-MoE GEMM kernel instead of tuning that kernel itself. See the [moe-configs section](#moe-configs) below.

**`--speculative-config '{"method":"mtp","num_speculative_tokens":3,"moe_backend":"triton"}'`** enables MTP (Multi-Token Prediction) speculative decoding using the model's own built-in prediction heads. These heads propose 3 candidate tokens per decode step from the hidden states already computed; vLLM verifies them all in a single batched forward pass. On coding workloads, measured acceptance is ~78% at position 0, ~56% at position 1, ~43% at position 2, yielding ~2.8 tokens per forward pass. The `moe_backend":"triton"` routes MoE expert computation through the same pre-tuned Triton kernel used for normal decoding. Earlier ngram speculative decoding (`--spec-method ngram --spec-tokens 5`) was removed — it crashed vLLM's `EngineCore` with a KV-cache block accounting assertion on large contexts; MTP uses a different code path and does not have this issue.

**`anthropic-shim`** is a separate Docker service running a stdlib-only Python HTTP server. It mounts `shim/shim.py` into the container and depends on `vllm-coding` being healthy. Nginx also depends on it (see the `depends_on` in the `nginx` service).

**`depends_on: condition: service_healthy`** means Nginx will not start until both vLLM and the shim pass their healthchecks. The 300s `start_period` gives vLLM time to load weights and compile CUDA graphs before its healthcheck begins polling.

**`MODEL_PATH` in the volume definition** reads from `.env`. If the variable is unset, it falls back to `/data/models`. The volume bind-mounts that host path into the container at `/models`, which is where the vLLM `--model` flag points.

### moe-configs

The `moe-configs/` directory holds pre-tuned Triton kernel configurations for vLLM's MoE (Mixture of Experts) GEMM kernels on the GB10 GPU. The filename encodes the tuning target:

```
E=256,N=512,device_name=NVIDIA_GB10,dtype=fp8_w8a8,block_shape=[128,128].json
```

Each file maps batch sizes (1, 2, 4, 8 … 4096) to optimal Triton tile parameters (`BLOCK_SIZE_M/N/K`, `GROUP_SIZE_M`, `num_warps`, `num_stages`). The filename is the lookup key — vLLM uses this config only when the kernel it selects for the model matches that key (expert count `E`, intermediate size `N`, `device_name`, `dtype`, and `block_shape`). When it matches, vLLM loads the pre-computed parameters instead of tuning the fused-MoE GEMM itself; when it doesn't, vLLM falls back to a generic config or its own tuning pass for that kernel.

Note this only covers the Triton fused-MoE GEMM. vLLM still runs other startup tuning passes regardless of this file (you'll see a FlashInfer `fp8_gemm` autotuner run in the logs), so it is not the only thing happening during the cold start.

These configs are hardware-specific: a config tuned for the GB10 will not perform well on an A100 or H100. If you are running on different hardware, either delete the `moe-configs/` directory or generate new configs using vLLM's tuning utilities.

---

## Step 4 — How Clients Reach the Model

A common pattern in local LLM stacks is to put a translation proxy (such as LiteLLM) in front of the inference server, because Claude Code speaks the **Anthropic** API (`POST /v1/messages`) while many inference servers historically spoke only the **OpenAI** API (`POST /v1/chat/completions`). The proxy converts one to the other.

**This stack needs a *lighter* layer:** a dependency-free, stdlib-only Python shim (`shim/shim.py`) that does one thing — hoists Claude Code's `system`-role messages from the `messages` array into vLLM's top-level `system` field. That's all it does; all other requests (and all responses, including SSE streams) pass through untouched.

Why is the shim needed? Claude Code injects MCP instructions, skills, and IDE context as a `role: "system"` message **inside** the `messages` array. vLLM's native `/v1/messages` endpoint strictly allows only `user` and `assistant` roles and 400s on `system` messages. The shim extracts those system messages, appends them to the `system` field, and forwards the corrected payload to vLLM — preserving the Anthropic response shape (including `thinking` blocks) that Claude Code expects.

The full request path:

```
Claude Code  (Anthropic /v1/messages) ─┐
                                        ├─→  Nginx :80 /coding/v1/messages  ─→  shim  ─→  vLLM :8000
OpenAI clients (OpenAI /v1/chat/...) ──┘
                                        └─→  Nginx :80 /coding/v1/chat/...  ─→  vLLM :8000
```

The shim listens on port 8080 inside the Docker network. Nginx routes `/coding/v1/messages` to the shim and everything else under `/coding/` to vLLM directly. OpenAI-compatible clients bypass the shim entirely because vLLM speaks OpenAI natively.

> **Why not use a full-featured translation proxy like LiteLLM?** A general-purpose proxy translates *between* Anthropic and OpenAI request schemas, which can be lossy for a reasoning model — an Anthropic→OpenAI→Anthropic round-trip may drop the model's `thinking` content blocks since the intermediate OpenAI schema represents reasoning differently. The shim only fixes the one thing that breaks (system-role messages) and preserves the full Anthropic response shape, which is why this stack uses a focused shim rather than a full translation proxy.

The only front-door component, then, is Nginx — covered next.

---

## Step 5 — Configure Nginx

Nginx is the single external-facing front door for the stack. It provides a clean path-based API: everything under `/coding/` routes to the coding model. Nginx uses two upstreams — `vllm_coding` (direct to vLLM) and `anthropic_shim` (to the normalizing shim) — and a more-specific location block so Claude Code's `/coding/v1/messages` path reaches the shim while all other OpenAI traffic goes straight to vLLM. This path-based routing is a deliberate design choice — if you later add a second model (reasoning, chat), you add a new upstream and a new location block without touching existing client configurations.

`nginx/nginx.conf`:

```nginx
worker_processes auto;
events { worker_connections 1024; }

http {
    upstream vllm_coding   { server vllm-coding:8000; }
    upstream anthropic_shim { server anthropic-shim:8080; }

    proxy_http_version      1.1;
    proxy_set_header        Host              $host;
    proxy_set_header        X-Real-IP         $remote_addr;
    proxy_set_header        X-Forwarded-For   $proxy_add_x_forwarded_for;
    proxy_set_header        Upgrade           $http_upgrade;
    proxy_set_header        Connection        "upgrade";
    proxy_read_timeout      600s;
    proxy_send_timeout      600s;
    client_max_body_size    50M;

    server {
        listen 80;
        server_name _;

        # /coding/v1/messages → shim (hoists Claude Code's system-role messages)
        location /coding/v1/messages {
            rewrite ^/coding(/.*)$ $1 break;
            proxy_pass http://anthropic_shim;
            proxy_buffering off;          # stream SSE tokens through immediately
        }

        # Everything else under /coding/ → vLLM directly
        location /coding/ {
            rewrite ^/coding(/.*)$ $1 break;
            proxy_pass http://vllm_coding;
            proxy_buffering off;          # stream SSE tokens through immediately
        }

        location /health {
            access_log off;
            return 200 "ok\n";
            add_header Content-Type text/plain;
        }

        location / {
            return 200 "DGX LLM API\nEndpoint: /coding/v1/\n";
            add_header Content-Type text/plain;
        }
    }

    # Uncomment and populate nginx/certs/ to enable TLS:
    # server {
    #     listen 443 ssl;
    #     server_name your-dgx-hostname.local;
    #
    #     ssl_certificate     /etc/nginx/certs/fullchain.pem;
    #     ssl_certificate_key /etc/nginx/certs/privkey.pem;
    #     ssl_protocols       TLSv1.2 TLSv1.3;
    #     ssl_ciphers         HIGH:!aNULL:!MD5;
    #
    #     location /coding/v1/messages { rewrite ^/coding(/.*)$ $1 break; proxy_pass http://anthropic_shim; proxy_buffering off; }
    #     location /coding/          { rewrite ^/coding(/.*)$ $1 break; proxy_pass http://vllm_coding; proxy_buffering off; }
    #     location /health            { access_log off; return 200 "ok\n"; add_header Content-Type text/plain; }
    # }
}
```

vLLM streams responses as **Server-Sent Events (SSE) over HTTP/1.1**, not WebSockets. Two settings make that stream flow token-by-token instead of arriving in one buffered clump:

- `proxy_http_version 1.1` keeps the upstream connection alive for the duration of the stream (HTTP/1.0 would close and break keep-alive). The `Upgrade`/`Connection` headers are harmless here and only matter if a client ever does negotiate an upgrade.
- `proxy_buffering off;` (set in the `/coding/` block) tells Nginx to forward each chunk to the client as it arrives rather than accumulating the response. Without it, Nginx buffers the upstream output and the client sees tokens land in bursts — the stream still completes, it just stops feeling incremental.

The 600s read and send timeouts apply *per read* — they cap how long Nginx waits between chunks from the upstream, not the total request duration. As long as a token streams at least every 600s the connection stays open, so even a very long generation completes fine: a 128K-token generation at ~60 tokens/second runs ~35 minutes end-to-end, far longer than 600s, yet never trips the timeout because tokens keep arriving. A 504 mid-stream therefore means the model stalled for >600s on a single token (an unusually large prompt prefill, for example); increase this value if you hit it.

The `/health` location provides a lightweight liveness endpoint for external monitors. The default `/` location returns a plain-text banner rather than a 404, which makes it easier to confirm the stack is reachable.

For TLS, uncomment the HTTPS server block and place your certificate chain at `nginx/certs/fullchain.pem` and private key at `nginx/certs/privkey.pem`. Let's Encrypt (via `certbot`) is the easiest source if the DGX has a resolvable hostname on your network.

> **Risk — No Authentication by Default:** This stack ships with **no access control**. Nginx forwards every request under `/coding/` unconditionally, and the served API key is `none`. Anyone who can reach port 80 (or 443) has full, unmetered use of the model — and since a coding assistant ingests your source files, that is both a compute-theft and a data-exposure surface. Treat the exposed ports as trusted-LAN-only. To require a key, add a check to the `/coding/` location, e.g.:
> ```nginx
> location /coding/ {
>     if ($http_authorization != "Bearer YOUR_LONG_RANDOM_SECRET") { return 401; }
>     rewrite ^/coding(/.*)$ $1 break;
>     proxy_pass http://vllm_coding;
>     proxy_buffering off;
> }
> ```
> Then set `ANTHROPIC_API_KEY` / `apiKey` to that secret in your clients. **Do not port-forward ports 80/443/8001 to the internet without this.**

> **Risk — No TLS by Default:** Port 80 is plaintext HTTP. If this machine is reachable on a network you don't fully control, add TLS certs and enable the HTTPS block. Without TLS, your API key and all model I/O (including the source code in your prompts) are exposed on the wire to anyone on the network segment.

---

## Step 6 — Launch the Stack

```bash
cd llm-stack
docker compose up -d
```

Startup sequence:

```
[1] vllm-coding starts
       ↓  loads NVFP4 weights into unified memory (~22 GB)
       ↓  allocates PagedAttention KV cache pool (~87 GB)
       ↓  compiles CUDA graphs for common batch sizes
       ↓  healthcheck passes at /health   ← takes 3–5 minutes
[2] anthropic-shim starts (waits for vllm-coding healthy)
       ↓  stdlib Python HTTP server on :8080
[3] nginx starts    (waits for vllm-coding + anthropic-shim healthy)
```

### What vLLM Is Actually Doing During Startup

The 3–5 minute startup is not just loading weights. vLLM is also:

1. **Allocating the KV cache:** PagedAttention divides the ~87 GB KV cache pool into fixed-size pages (blocks). This allocation must happen upfront so that concurrent requests can be scheduled deterministically. The `--gpu-memory-utilization 0.85` flag tells vLLM to commit 85% of the 128 GB unified memory pool for weights + KV cache combined.
2. **Compiling CUDA graphs:** For common input shapes (batch size 1, 2, 4, 8, etc.), vLLM pre-compiles optimized CUDA execution graphs. At inference time, the graph is replayed rather than re-dispatched, reducing per-token latency significantly.

Monitor startup:

```bash
docker compose logs -f vllm-coding    # Watch for "Application startup complete"
docker compose logs -f nginx
```

> **Risk — Cold Start Time:** The stack will appear unresponsive for several minutes during startup. Nginx blocks on vLLM's healthcheck — this is correct behavior.

> **Risk — restart: "no":** The compose file does not auto-restart containers on crash. If vLLM OOMs or hits a CUDA error, you must manually run `docker compose up -d` again. Change to `restart: unless-stopped` for production deployments.

---

## Step 7 — Verify & Connect Clients

### How the Inference Optimizations Work Together

Before connecting clients, it's worth understanding the three inference optimizations in this stack and how they interact, because their benefits compound:

**Prefix caching** stores the computed KV tensors for prompt prefixes that have been seen before. Claude Code always sends the same large system prompt (several thousand tokens of instructions about tools, file formats, and behavior). Without prefix caching, every request recomputes those KV tensors from scratch — wasting compute proportional to the system prompt length. With prefix caching enabled, the first request pays the full cost, and every subsequent request in the same session skips the system prompt computation entirely. For Claude Code's usage pattern (many requests per session, identical system prompt), this is one of the highest-impact optimizations in the stack.

**MTP speculative decoding** exploits a property of code generation: code is highly repetitive and locally predictable. The Qwen3.6-35B-A3B-NVFP4 checkpoint includes built-in Multi-Token Prediction heads that propose 3 candidate tokens per decode step using hidden states already computed for the current position — no separate draft model required. vLLM verifies all 3 candidates in a single batched forward pass through the full target model. When candidates are accepted (frequently, for boilerplate patterns like `return None`, `self.`, identifiers just introduced), the sequence advances by 2–3 tokens for the cost of 1 forward pass. Measured on coding workloads: ~2.8 tokens per forward pass on average, yielding ~100–110 tok/s vs. the ~60–80 tok/s baseline without speculative decoding.


**PagedAttention** manages the KV cache as virtual memory pages rather than pre-allocating contiguous blocks per sequence. This is what allows `--max-num-seqs 32` — 32 concurrent requests in flight — without each one reserving 128K × KV cache bytes upfront. Pages are allocated on demand as context grows, and released immediately when a sequence completes. This matters for interactive coding use: when Claude Code has 10 open editor contexts, each making occasional requests, PagedAttention ensures they share the KV cache pool efficiently rather than starving each other.

```
Prefix cache hit:   Skip recomputing large system prompt    → lower latency
PagedAttention:     32 concurrent seqs sharing KV pool     → better utilization
```

These work together: prefix caching reduces the effective prompt length for scheduling purposes (the biggest win for Claude Code's repeated system prompt), and PagedAttention ensures the KV cache pool is shared efficiently across concurrent sequences.

### Quick Smoke Test

```bash
# Direct vLLM (bypasses Nginx — useful for debugging)
curl http://localhost:8001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"coding","messages":[{"role":"user","content":"Hello"}],"max_tokens":50}'

# Through Nginx, OpenAI format (this is what Continue and other OpenAI clients send)
curl http://localhost/coding/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"coding","messages":[{"role":"user","content":"Hello"}],"max_tokens":50}'

# Through Nginx, Anthropic format (this is what Claude Code sends — routed through the shim)
curl http://localhost/coding/v1/messages \
  -H "Content-Type: application/json" \
  -H "anthropic-version: 2023-06-01" \
  -d '{"model":"coding","messages":[{"role":"user","content":"Hello"}],"max_tokens":50}'
```

### Connect Claude Code

```bash
# Add to ~/.bashrc or ~/.zshrc — replace <dgx-ip> with the DGX Spark's address
# (use localhost if Claude Code runs on the DGX itself)
alias claude-local='ANTHROPIC_BASE_URL=http://<dgx-ip>/coding \
  ANTHROPIC_API_KEY=none \
  ANTHROPIC_DEFAULT_OPUS_MODEL=coding \
  ANTHROPIC_DEFAULT_SONNET_MODEL=coding \
  ANTHROPIC_DEFAULT_HAIKU_MODEL=coding \
  claude'
```

`ANTHROPIC_BASE_URL` points at Nginx's `/coding` path; Claude Code appends `/v1/messages`, which Nginx routes through the normalizing shim before reaching vLLM. The shim hoists Claude Code's system-role messages into the top-level `system` field so vLLM doesn't reject them with a 400. The `ANTHROPIC_*_MODEL` vars map Claude Code's internal model references (opus/sonnet/haiku) onto the served model name `coding`. Use `claude-local` to route Claude Code through the local stack, `claude` to use Anthropic's cloud API. Switching requires a Claude Code restart — `ANTHROPIC_BASE_URL` is read once at startup.

### Connect Continue (VS Code / JetBrains)

In `~/.continue/config.yaml`:

```yaml
models:
  - name: Qwen3.6 Coding (local)
    provider: openai
    model: coding
    apiBase: http://<dgx-ip>/coding/v1
    apiKey: none
    systemMessage: "/no_think"
    roles:
      - chat
      - edit
```

The `systemMessage: "/no_think"` is required. The model reasons by default; vLLM extracts thinking tokens into a `reasoning_content` delta field that Continue's OpenAI adapter does not handle, producing a connection error. The `/no_think` directive suppresses reasoning for Continue's requests. The `apiBase` must use port 80 (the nginx path) — the vLLM debug port `:8001` is bound to loopback only and is not reachable from a remote client.

---

## Port Reference

| Port | Service | Format | Use |
|---|---|---|---|
| `:8001` | vLLM direct | OpenAI + Anthropic | Internal / debug only — **bound to `127.0.0.1`, not network-reachable** |
| `:80` | Nginx | OpenAI + Anthropic | Claude Code (`claude-local`), Continue, curl |
| `:443` | Nginx TLS | OpenAI + Anthropic | Same as above, encrypted (optional) |

---

## Throughput Expectations and Limitations

**Latency (time to first token):** On a warm cache hit (second request in a session with same system prompt), first-token latency is typically 1–3 seconds. Cold requests (new session, full system prompt recomputation) take 3–8 seconds depending on prompt length.

**Throughput (tokens/second):** Expect 100–110 tokens/second on typical coding responses with MTP speculative decoding active — a 500-token response streams in under 6 seconds. Acceptance rate varies by workload: structured code (high repetition, predictable identifiers) sits at ~78% for the first draft token and ~43% for the third, yielding ~2.8 tokens per target-model forward pass. Unstructured prose is lower. The baseline without speculative decoding is ~60–80 tok/s.

**Context window:** The stack is configured for 128K tokens (`--max-model-len 131072`). In practice, Claude Code rarely exceeds 32K tokens per request. The full 128K is available if needed, but very long contexts increase KV cache consumption and reduce how many concurrent sequences can be served simultaneously.

**Concurrency ceiling:** `--max-num-seqs 32` allows 32 in-flight requests. For a single developer this is never the bottleneck. For a small team sharing the stack, it provides comfortable headroom. The real ceiling is KV cache exhaustion: if all 32 sequences are holding large contexts simultaneously, PagedAttention will begin queuing new requests.

**The honest comparison to cloud:** A local stack trades latency predictability for cost and privacy. Cloud APIs serve requests from large GPU clusters and can burst to higher throughput. This stack will have more variance — a fresh model load or an unusually long context can cause noticeable pauses. The benefit is that every token generated stays on your hardware.

---

## Risk Summary

| # | Risk | Severity | Mitigation |
|---|---|---|---|
| 1 | Driver < 580.x | Critical | Check `nvidia-smi` before building — NVFP4 requires Blackwell driver |
| 2 | vLLM < 0.19 | Critical | Build from `eugr/spark-vllm-docker` — NVFP4 kernel support added in 0.19 |
| 3 | FP8 KV cache flag added | High | Do NOT add `--kv-cache-dtype fp8` — checkpoint lacks calibration scales, causes silent value clipping |
| 4 | No authentication | High | API is wide open to anyone who can reach the port — add an Nginx bearer-token check before exposing beyond a trusted LAN; never port-forward without it |
| 5 | No TLS on Nginx | High | Add certs if not on a private LAN — all tokens, API keys, and prompt source code are plaintext on :80 |
| 6 | `restart: "no"` | Medium | Change to `unless-stopped` for production — OOM or CUDA error requires manual recovery |
| 7 | Streaming timeouts | Medium | Tune `proxy_read_timeout` upward if you see 504 errors on long generations |
| 8 | Cold start ~5 min | Low | Wait for healthcheck before testing — the stack is not broken, it's loading |
