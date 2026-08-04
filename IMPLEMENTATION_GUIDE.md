# DGX Spark vLLM Stack — Implementation Guide

## Why This Stack Exists

Running large language models locally on developer hardware has historically meant accepting a painful tradeoff: models small enough to fit in GPU VRAM are too weak for serious coding work, while models capable enough require multi-GPU server racks. This stack exists because the DGX Spark changes that equation — and takes some deliberate effort to unlock properly.

The architecture here has one goal: make a 35-billion-parameter reasoning model feel like a cloud API, fast enough for interactive use, running entirely on a single machine you own.

---

## Architecture Overview

```
  Hermes Agent            Claude Code / IDEs        curl / Continue
  (PRIMARY, on-host)      (claude-local alias)      (OpenAI-compatible)
  OpenAI API                     │                         │
        │              :80 /coding (Anthropic)   :80 /coding/v1 (OpenAI)
        │                        └───────────┬─────────────┘
        │                                    ▼
        │                    ┌───────────────────────────────────┐
        │                    │        Nginx  :80 / :443          │
        │                    │  /coding/v1/messages → shim:8080  │
        │                    │  /coding/*           → vLLM:8000  │
        │                    └────────┬────────────────┬─────────┘
        │                             ▼                │
        │                  ┌─────────────────────┐     │
        │                  │  anthropic-shim     │     │
        │                  │  hoists system[]    │     │
        │                  └──────────┬──────────┘     │
        │  :8001 direct               │                │
        │  bypasses Nginx + shim      ▼                ▼
        └──────────────────►┌──────────────────────────────────┐
                            │  vLLM — coding                   │
                            │  Qwen3.6-35B-A3B-NVFP4 · 256K    │
                            │  DFlash spec decode (4)          │
                            │  111 tok/s c=1 · 228 tok/s c=4   │
                            └────────────────┬─────────────────┘
                                             ▼
                            ┌──────────────────────────────────┐
                            │  NVIDIA container runtime        │
                            │  CUDA · Marlin · PagedAttention  │
                            └────────────────┬─────────────────┘
                                             ▼
                            ┌──────────────────────────────────┐
                            │  GB10 Grace Blackwell            │
                            │  128 GB unified · 273 GB/s       │
                            │  memory-bound at concurrency 1–8 │
                            └──────────────────────────────────┘
```

### GPU Memory Budget (~121 GiB usable, gpu_util = 0.75 → ~91 GiB committed)

```
[███████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░]
 19%  22.6 GiB     57%  68.6 GiB KV + activations    25% free
 NVFP4 weights     (PagedAttention pool)             ~30 GiB
 (21.9 main +
  0.7 draft)
```

Measured live: `nvidia-smi` reports **93,381 MiB (91.2 GiB)** for `VLLM::EngineCore`. vLLM's
own `cache_config_info` metric reports a KV pool of **2,453,340 tokens** and
**`kv_cache_max_concurrency` = 9.36** — that is, room for roughly nine simultaneous
sequences at the full 262144-token context.

> **Why 0.75 and not a higher fraction.** Utilization was lowered from 0.85 deliberately.
> This is a unified-memory machine: the same pool serves the OS, Docker, and every other
> process on the box, and its consumption grows over days of uptime. At 0.85 the host sat at
> ~122 GB of 128 GB, leaving no margin for a slow leak elsewhere to avoid an OOM. Utilization
> buys KV capacity, **not throughput** — so the ~30 GiB given up here costs nothing in
> tokens/second and buys the ability to leave the box running unattended.

### Clients — and why Hermes takes a different path

Three kinds of client reach this stack, and they do not all go through the front door.

**Hermes Agent is the primary workload**, and it is the reason the tuning in
[OPTIMIZATION_REPORT.md](OPTIMIZATION_REPORT.md) was done at all. It runs a general personal
assistant, a small project builder, and content development. It connects **directly to
`http://localhost:8001/v1`** — vLLM's published loopback port — using the OpenAI API, and
**bypasses both Nginx and the shim.**

That is deliberate, not an oversight. Nginx exists to terminate network traffic and to route
Anthropic-format requests to the normalizing shim. Hermes needs neither: it runs on the same
host, so there is nothing to reverse-proxy, and it speaks the OpenAI API natively, so there
is nothing to normalize. Routing it through Nginx would add two process hops per request for
no benefit.

Two consequences worth stating explicitly:

- **Port 8001 must stay bound to `127.0.0.1`.** It is unauthenticated and unencrypted.
  Because Hermes depends on it, it is tempting to "just expose it" for a second machine —
  don't. Anything that can reach it gets unrestricted use of the model and sees every prompt.
  A remote client should go through Nginx and get an auth check added first.
- **Hermes reads the real context window from `/v1/models`.** This is the opposite of Claude
  Code, which assumes ~200K for any custom endpoint and cannot be told otherwise — the
  problem that forced `--max-model-len 262144`. A client that probes adapts to whatever the
  server offers; a client that assumes constrains the server's configuration.

**Availability.** `restart: "no"` is intentional (see the risk table) so that a crash stays
visible instead of silently restart-looping. To stop that from leaving the assistant with no
model, Hermes is configured with an **AWS Bedrock fallback provider**, which fires only on
rate-limit, overload, or connection failure against the local stack. It is billed per token,
so it should stay a genuine fallback rather than a routine path.

**Why the workload shape mattered.** Hermes traffic is dominated by structured tool calls and
file edits rather than free-form prose. That single fact selected the speculative decoding
method: DFlash wins on predictable output and loses slightly on novel text. A prose-heavy
deployment on identical hardware should make the opposite choice.

---

## The Platform: Why the DGX Spark Changes the Rules

Traditional GPU-based inference has a hard constraint: model weights must fit in VRAM, and VRAM is a small, expensive, physically separate pool — typically 24–80 GB on a workstation. On those machines, a 35B parameter model in full precision (BF16) requires ~70 GB and simply doesn't fit. Even with aggressive quantization you're pushing limits, and a KV cache large enough for 128K context is out of the question.

The DGX Spark's GB10 Grace Blackwell changes this with unified memory. The CPU and GPU do not have separate memory pools — there is a single 128 GB pool addressed by both. This is not a software trick like CPU offloading (which shuttles weights across PCIe); it is the hardware architecture, and the GPU's tensor cores address the entire 128 GB directly.

The advantage here is **capacity, not raw bandwidth.** The unified pool runs at ~273 GB/s (LPDDR5X) — modest next to a discrete GPU's VRAM (an H100 moves >3 TB/s). What the Spark buys you is the ability to hold a 35B model *and* a large KV cache in one pool at all, on a single device, without the multi-GPU rack that capacity would otherwise demand. You trade peak throughput for the fact that the workload fits — which is why generation lands at ~110 tokens/second (see [Throughput Expectations](#throughput-expectations-and-limitations)) rather than cloud-cluster speeds.

What this means in practice:
- A 35B model fits with room to spare for a large KV cache
- The KV pool holds ~2.45 M tokens — about **nine** concurrent conversations at the full
  262144-token context, or proportionally more at shorter contexts
- There is no PCIe copy between CPU and GPU memory — both address the same physical pool

The GB10 supports NVFP4 quantization, which is what the `Qwen3.6-35B-A3B-NVFP4` checkpoint uses. The primary benefit is **memory**: weights stored in 4-bit take ~22 GB instead of ~70 GB at BF16 — this is what makes the full 35B model fit alongside the large KV cache. On the compute side, the current vLLM build runs those weights via the **Marlin weight-only path**: weights stay in 4-bit in memory (the size advantage is fully preserved), but at inference time they are dequantized to BF16 before the matrix multiply, which runs on standard BF16 tensor cores.

**This distinction was tested directly, and it does not matter here.** A checkpoint quantized `W4A16_NVFP4` — 4-bit weights, 16-bit activations — *cannot* drive FP4 tensor cores on any GPU, because FP4 tensor cores compute FP4×FP4 and 16-bit activations force the multiply back to 16-bit math. That is the Marlin path by definition. So vLLM's startup warning that "your GPU does not have native support for FP4 computation" is misleading: GB10 *does* have FP4 tensor cores; this checkpoint simply cannot feed them.

We obtained a checkpoint that can (`W4A4`, 4-bit weights *and* activations) and benchmarked native FP4 against Marlin properly. The result was a **wash** (104.1 vs 101.3 tok/s at concurrency 1), and the W4A4 checkpoint was ~16% slower overall because it is 3 GB larger. The reason is in [Throughput Expectations](#throughput-expectations-and-limitations): at low concurrency this machine is **memory-bound**, so the cost of reading weights dominates and the speed of the multiply is nearly irrelevant. Measured throughput with the deployed setup is **110–111 tokens/second** — a 500-token response streams in under 5 seconds.

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
- **`moe-configs/`**: retained for reference but **no longer mounted** — see [moe-configs](#moe-configs) for why. If you re-enable it for a different model, verify the filename's `dtype=` matches the checkpoint's actual dtype, or it will silently never apply.
- **Speculative decoding**: `--speculative-config` is model-specific. The DFlash draft model is trained for Qwen3.6 and will not work with anything else. Re-sweep `num_speculative_tokens` for any new model — the optimum is workload- and hardware-dependent, and vendor defaults were badly wrong here.
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

> **Risk — Image Tag:** The `docker-compose.yml` hardcodes `vllm-node-v2:latest`. If you tag the image differently during the build, update the compose file before proceeding.

Verify the build succeeded and vLLM reports the correct version:

```bash
docker run --rm --runtime=nvidia vllm-node-v2:latest python3 -c "import vllm; print(vllm.__version__)"
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

The KV cache stores the computed key/value attention tensors for every token in every active context window. At BF16 (the default), the KV pool plus activations occupies ~68.6 GiB in this stack. FP8 quantization of the KV cache would roughly double its token capacity — a large gain.

However, FP8 KV cache quantization requires per-tensor calibration scales (`k_scale`, `v_scale`, `q_scale`) that tell vLLM how to rescale the values before quantizing them. These scales must be computed on a representative dataset and baked into the checkpoint. **The `Qwen3.6-35B-A3B-NVFP4` checkpoint does not include these scales.**

When scales are missing, vLLM falls back to `scale=1.0` — meaning no rescaling. FP8 E4M3 can only represent values up to 448. Any attention tensor value above 448 is silently clipped. In practice this means the model computes attention over subtly wrong values on long contexts: the kind of errors that produce wrong variable names, hallucinated API signatures that look plausible, or off-by-one logic that's hard to catch in review. This is exactly the wrong failure mode for a coding assistant.

The decision is: leave KV cache at BF16, accept the full cost, and preserve accuracy. **Do not add `--kv-cache-dtype fp8` to the vLLM command with this checkpoint.**

Two later findings that refine — but do not overturn — this conclusion:

- **A different checkpoint does ship the scales.** The `unsloth/Qwen3.6-35B-A3B-NVFP4` variants include calibrated `k_scale`/`v_scale` tensors, and with them FP8 KV cache was measured to be safe (long-context recall passed) and to cost **0% throughput**. So the objection above is specific to *this* checkpoint's missing calibration, not to FP8 KV cache as a technique.
- **It buys capacity, not speed.** Even where it works, FP8 KV changed throughput by nothing measurable. It is worth adopting only if KV capacity becomes the binding constraint. At `--gpu-memory-utilization 0.75` with ~9 concurrent full-context sequences available and ~30 GiB spare, it is not — which is why the faster NVIDIA checkpoint was kept over the unsloth one despite the latter's working FP8 KV support.

**Always verify before enabling:** check that the checkpoint actually contains `k_scale`/`v_scale` tensors rather than assuming, because the failure mode is silent.

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
    image: vllm-node-v2:latest  # Built locally via eugr/spark-vllm-docker — must be vLLM ≥0.19 for NVFP4
    container_name: vllm-coding
    runtime: nvidia
    restart: "no"
    networks:
      - llm-net
    volumes:
      - models:/models
      - huggingface-cache:/root/.cache/huggingface
      # ./moe-configs is deliberately NOT mounted — see the moe-configs section
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
      - HF_TOKEN=${HF_TOKEN:-}
      - VLLM_LOGGING_LEVEL=WARNING
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
      - "262144"
      - "--gpu-memory-utilization"
      - "0.75"
      - "--tensor-parallel-size"
      - "1"
      - "--enable-prefix-caching"
      - "--max-num-seqs"
      - "32"
      - "--max-num-batched-tokens"
      - "8192"
      - "--reasoning-parser"
      - "qwen3"
      - "--enable-auto-tool-choice"
      - "--tool-call-parser"
      - "qwen3_coder"
      - "--language-model-only"    # Skip vision encoder — not needed for coding, frees VRAM
      - "--speculative-config"
      - '{"method":"dflash","model":"/models/Qwen3.6-35B-A3B-DFlash","num_speculative_tokens":4}'
      # NOTE: ngram speculative decoding was removed — it crashed EngineCore on
      # vLLM 0.22.1rc1, and when later retested scored 47.8 tok/s at 17.2%
      # acceptance vs 123.7 for a real draft model.
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
      start_period: 600s   # measured cold start is 341–461s; see note below

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

**`--max-num-batched-tokens 8192`** raises the scheduler's per-step token budget from the implicit default of 2048. vLLM warns explicitly that the default starves speculative-decoding draft slots. Honest caveat: the measured gain sits **at the noise floor and is not established** — it is kept because it is the documented-correct setting for a spec-decode configuration, not because we proved it faster.

**`--speculative-config '{"method":"dflash","model":"/models/Qwen3.6-35B-A3B-DFlash","num_speculative_tokens":4}'`** enables speculative decoding with **DFlash**, a separate 0.77 GB six-layer dense draft model. The draft model proposes 4 candidate tokens per step; vLLM verifies all 4 in a single batched forward pass through the full target model. When candidates are accepted the sequence advances several tokens for the cost of one forward pass.

This is **the single largest tunable win in the stack: +44%** over no speculation (111.4 vs 77.3 tok/s at concurrency 1). Two non-obvious points, both established by measurement in [OPTIMIZATION_REPORT.md](OPTIMIZATION_REPORT.md):

- **The draft length was swept, not copied.** The upstream recipe for a different quantization of this model uses 15. On this hardware 15 was the *worst* value tested, and at concurrency 4 it was **slower than disabling speculation entirely**. The accepted-token count is roughly constant regardless of draft length — the draft model only ever gets 2–3 tokens right, so everything beyond that is verification cost with no return.
- **DFlash was chosen over MTP because of the workload.** The checkpoint ships built-in Multi-Token Prediction heads, which need no separate draft model. MTP is a perfectly good choice — it simply loses here on tool calls (−7.2%) and concurrency (−9.1%) while winning slightly on prose. Speculation pays off in proportion to how predictable the output is, and agentic traffic is dominated by structured tool calls and code edits. **A prose-heavy deployment should prefer MTP:** `'{"method":"mtp","num_speculative_tokens":3,"moe_backend":"triton"}'` — note that MTP's spec config *requires* `"moe_backend":"triton"`, because its draft layer is unquantized BF16 and omitting the field crashes `EngineCore`.

**`anthropic-shim`** is a separate Docker service running a stdlib-only Python HTTP server. It mounts `shim/shim.py` into the container and depends on `vllm-coding` being healthy. Nginx also depends on it (see the `depends_on` in the `nginx` service).

**`depends_on: condition: service_healthy`** means Nginx will not start until both vLLM and the shim pass their healthchecks. The `start_period` gives vLLM time to load weights and compile CUDA graphs before the healthcheck begins counting failures.

> **Why 600s and not 300s.** Cold start was measured across ten container recreates at **341–461 seconds**. The old 300s `start_period` plus 5 retries × 30s gave 450s of total grace — *below an observed load time*. A slow start would flip the container to `unhealthy`, and because both Nginx and the shim gate on `service_healthy`, the whole stack would stall behind it. Set this from measured load time, not a guess.

**`MODEL_PATH` in the volume definition** reads from `.env`. If the variable is unset, it falls back to `/data/models`. The volume bind-mounts that host path into the container at `/models`, which is where the vLLM `--model` flag points.

### moe-configs

> **Status: retained for reference, deliberately NOT mounted.** The directory and its
> `VLLM_TUNED_CONFIG_FOLDER` environment variable were both removed from
> `docker-compose.yml`. The explanation below is kept because the mechanism is worth
> understanding — and because the reason it was removed is a useful cautionary tale.

The `moe-configs/` directory holds pre-tuned Triton kernel configurations for vLLM's MoE (Mixture of Experts) GEMM kernels on the GB10 GPU. The filename encodes the tuning target:

```
E=256,N=512,device_name=NVIDIA_GB10,dtype=fp8_w8a8,block_shape=[128,128].json
```

Each file maps batch sizes (1, 2, 4, 8 … 4096) to optimal Triton tile parameters (`BLOCK_SIZE_M/N/K`, `GROUP_SIZE_M`, `num_warps`, `num_stages`). The filename is the lookup key — vLLM uses this config only when the kernel it selects for the model matches that key (expert count `E`, intermediate size `N`, `device_name`, `dtype`, and `block_shape`). When it matches, vLLM loads the pre-computed parameters instead of tuning the fused-MoE GEMM itself; when it doesn't, vLLM falls back to a generic config or its own tuning pass for that kernel.

**Why it was removed.** Read the filename again: `dtype=fp8_w8a8`. This model is **NVFP4**, not FP8. The key never matched, so the config was never loaded — it had been inert since the day it was added, and every "it's helping" assumption about it was wrong.

Renaming it to what vLLM actually looks for made things worse, not better: the config loaded and then crashed the engine outright with

```
triton.runtime.errors.OutOfResources: shared memory,
Required: 294912, Hardware limit: 101376
```

The tile parameters were tuned for a GPU with roughly 3× GB10's shared memory per SM. **The filename mismatch had been silently protecting the stack from a config that would have crashed it.** Leaving the directory mounted is therefore a latent hazard: if a future vLLM release loosens its matching rules, a file that does nothing today becomes a file that takes the server down.

If you want a genuinely tuned MoE config for this hardware, generate one with vLLM's own `benchmark_moe.py` against *this* GPU and *this* dtype. Do not reuse one from another machine.

Note this only ever covered the Triton fused-MoE GEMM. vLLM still runs other startup tuning passes regardless (you'll see a FlashInfer `fp8_gemm` autotuner run in the logs), so it was never the only thing happening during the cold start.

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
       ↓  allocates PagedAttention KV cache pool (~68.6 GiB, 2.45 M tokens)
       ↓  compiles CUDA graphs for common batch sizes
       ↓  healthcheck passes at /health   ← takes 3–5 minutes
[2] anthropic-shim starts (waits for vllm-coding healthy)
       ↓  stdlib Python HTTP server on :8080
[3] nginx starts    (waits for vllm-coding + anthropic-shim healthy)
```

### What vLLM Is Actually Doing During Startup

The 3–5 minute startup is not just loading weights. vLLM is also:

1. **Allocating the KV cache:** PagedAttention divides the ~68.6 GiB KV cache pool into fixed-size pages (blocks). This allocation must happen upfront so that concurrent requests can be scheduled deterministically. The `--gpu-memory-utilization 0.75` flag tells vLLM to commit 75% of the unified memory pool for weights + KV cache combined — ~91.2 GiB measured, holding 2,453,340 KV tokens.
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

**DFlash speculative decoding** exploits a property of agentic and code generation: the output is highly repetitive and locally predictable. A separate 0.77 GB six-layer draft model proposes 4 candidate tokens per decode step; vLLM verifies all 4 in a single batched forward pass through the full target model. When candidates are accepted (frequently, for JSON tool-call scaffolding, boilerplate like `return None`, and identifiers just introduced), the sequence advances several tokens for the cost of one forward pass.

**This is the largest tunable win in the stack.** Measured across four representative workloads: **111.4 tok/s vs 77.3 without speculation — +44%** at concurrency 1, and 227.8 vs 181.3 (+26%) at concurrency 4, at ~53% draft-token acceptance.

Why it matters so much here specifically: at low concurrency this box is **memory-bound**, spending most of its time reading weights rather than computing. Speculation is the one lever that produces more output tokens per weight-read, which is exactly the constrained resource. That is also why kernel-level optimizations produced nothing measurable — see [OPTIMIZATION_REPORT.md](OPTIMIZATION_REPORT.md).


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

**Throughput (tokens/second):** Measured with DFlash speculative decoding active, medians over n=14 trials per workload:

| workload | tok/s @ c=1 |
|---|---|
| code edits | 121.0 |
| tool calls (structured JSON) | 118.7 |
| general reasoning | 104.1 |
| novel prose | 100.1 |
| **overall (median of the above)** | **111.4** |
| *no speculative decoding* | *77.3* |

Aggregate throughput at 4 concurrent streams is **227.8 tok/s**; draft acceptance is ~53%.
A 500-token response streams in under 5 seconds.

**Note the spread between workloads is larger than most configuration deltas.** Speculation pays off in proportion to how predictable the output is, so structured tool calls run ~20% faster than novel prose on identical settings. Benchmark on prompts that resemble your actual traffic — a single unrepresentative prompt can invert a conclusion.

**This is memory-bound, not compute-bound.** Roughly 0.9 GB of weights is read per token against a ~273 GB/s ceiling, putting the machine at about a quarter of peak bandwidth. A sustained single-stream run showed 93% reported GPU utilization at **26.7 W** — stalled on memory, not computing. This is why speculative decoding (more tokens per weight-read) is the dominant lever and why kernel-level changes measured as noise.

**Context window:** The stack is configured for **256K tokens** (`--max-model-len 262144`). This is deliberately larger than needed. Claude Code assumes a ~200K window for any custom endpoint and has no way to learn the backend's real limit; with the previous 131072 setting it would compose requests that vLLM hard-rejected with `prompt + max_tokens > max-model-len` before generation began, surfacing to the user as a spinner that never resolved. Setting the server's window *above* what the client assumes makes the client's own auto-compaction fire safely below the real wall. The extra headroom is nearly free: only 10 of the model's 40 layers use a growing KV cache (~20 KiB/token), so doubling the window cost about 2.7 GB per maximally-long sequence, not a doubling of memory.

**Concurrency ceiling:** `--max-num-seqs 32` allows 32 in-flight requests, but the real ceiling is KV capacity. vLLM reports a pool of **2,453,340 tokens** and `kv_cache_max_concurrency` of **9.36** — about nine simultaneous sequences at the *full* 262144-token context, proportionally more at realistic lengths. For a single user this is never the bottleneck; for a small team, watch this metric rather than `--max-num-seqs`.

**The honest comparison to cloud:** A local stack trades latency predictability for cost and privacy. Cloud APIs serve requests from large GPU clusters and can burst to higher throughput. This stack will have more variance — a fresh model load or an unusually long context can cause noticeable pauses. The benefit is that every token generated stays on your hardware.

---

## Risk Summary

| # | Risk | Severity | Mitigation |
|---|---|---|---|
| 1 | Driver < 580.x | Critical | Check `nvidia-smi` before building — NVFP4 requires Blackwell driver |
| 2 | vLLM < 0.19 | Critical | Build from `eugr/spark-vllm-docker` — NVFP4 kernel support added in 0.19 |
| 3 | FP8 KV cache flag added | High | Do NOT add `--kv-cache-dtype fp8` with *this* checkpoint — it lacks `k_scale`/`v_scale`, causing silent value clipping. Verify the scales exist before enabling on any checkpoint |
| 4 | No authentication | High | API is wide open to anyone who can reach the port — add an Nginx bearer-token check before exposing beyond a trusted LAN; never port-forward without it |
| 5 | No TLS on Nginx | High | Add certs if not on a private LAN — all tokens, API keys, and prompt source code are plaintext on :80 |
| 6 | `restart: "no"` | Medium | Change to `unless-stopped` for production — OOM or CUDA error requires manual recovery. Kept as `no` here so crashes stay visible rather than silently restart-looping; the client-side fallback provider covers availability |
| 7 | Streaming timeouts | Medium | Tune `proxy_read_timeout` upward if you see 504 errors on long generations |
| 8 | Cold start 6–8 min | Low | Measured 341–461s across 10 recreates. Wait for the healthcheck — the stack is not broken, it's loading. `start_period` must exceed your slowest observed load |
