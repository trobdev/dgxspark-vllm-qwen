# DGX Spark vLLM Stack — Implementation Guide

This is the "why" document: how the stack fits together, and the reasoning behind each
configuration choice. For step-by-step installation, see [SETUP_GUIDE.md](SETUP_GUIDE.md).
For the measurements behind the tuning, see [TUNING.md](TUNING.md).

## Why This Stack Exists

Running large language models locally on developer hardware has historically meant
accepting a painful tradeoff. Models small enough to fit in GPU VRAM are too weak for
serious coding work, while capable models need multi-GPU server racks. The DGX Spark
changes that, but getting the most out of it takes deliberate effort.

The architecture here has one goal: make a 35-billion-parameter reasoning model feel like a
cloud API, fast enough for interactive use, running entirely on a single machine you own.

---

## Architecture Overview

```
  On-host agent             Claude Code / IDEs        curl / Continue / SDKs
  (scripts, frameworks)     (claude-local alias)      (OpenAI-compatible)
  OpenAI API                       │                         │
        │                :80 /coding (Anthropic)   :80 /coding/v1 (OpenAI)
        │                          └───────────┬─────────────┘
        │                                      ▼
        │                    ┌───────────────────────────────────┐
        │                    │  Nginx  :80   (API key required)  │
        │                    │  /coding/v1/messages → shim:8080  │
        │                    │  /coding/*           → vLLM:8000  │
        │                    └────────┬────────────────┬─────────┘
        │                             ▼                │
        │                  ┌─────────────────────┐     │
        │                  │  anthropic-shim     │     │
        │                  │  hoists system[]    │     │
        │                  └──────────┬──────────┘     │
        │  127.0.0.1:8001 direct      │                │
        │  bypasses Nginx + shim      ▼                ▼
        └──────────────────►┌──────────────────────────────────┐
                            │  vLLM — coding                   │
                            │  Qwen3.6-35B-A3B-NVFP4 · 256K    │
                            │  DFlash spec decode (4)          │
                            │  112 tok/s c=1 · 235 tok/s c=4   │
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

| Service | Image | Published port | Role |
|---|---|---|---|
| `vllm-coding` | `vllm-node-v2:latest` (built locally) | `127.0.0.1:8001` | The model server; speaks OpenAI and Anthropic APIs |
| `anthropic-shim` | `python:3.12-slim` + [`shim/shim.py`](shim/shim.py) | `127.0.0.1:8003` | Normalizes Claude Code's requests for vLLM |
| `nginx` | `nginx:alpine` + [`nginx/default.conf.template`](nginx/default.conf.template) | `80` (all interfaces) | Network front door; checks the API key; routes by path |

### GPU Memory Budget (~121 GiB usable, gpu_util = 0.75 → ~91 GiB committed)

```
[███████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░]
 19%  22.6 GiB     57%  68.6 GiB KV + activations    25% not committed
 NVFP4 weights     (PagedAttention pool)             to vLLM, ~30 GiB
 (21.9 main +                                        (OS, Docker, other
  0.7 draft)                                          processes share it)
```

- **Measured live:** `nvidia-smi --query-compute-apps` reports **93,381 MiB (91.2 GiB)** for
  `VLLM::EngineCore`.
- **KV pool:** vLLM's `cache_config_info` metric reports **2,453,340 tokens** and
  **`kv_cache_max_concurrency` = 9.36**, room for roughly nine simultaneous sequences at the
  full 262144-token context.

> **Why 0.75 and not a higher fraction.** Utilization was lowered from 0.85 deliberately.
> This is a unified-memory machine: the same pool serves the OS, Docker, and every other
> process on the box, and their usage grows over days of uptime. At 0.85 the whole box sat at
> ~122 GB of 128 GB, with no margin if something else leaked memory slowly. At 0.75, ~17 GiB
> of host memory stays free. Utilization buys KV capacity, **not throughput**, so giving it
> up costs nothing in tokens per second and lets the box run unattended.

### Clients — and why on-host agents take a different path

Three kinds of client reach this stack, and they don't all go through the front door.

**On-host agents connect directly to `http://localhost:8001/v1`**, vLLM's loopback-only
port, using the OpenAI API. They bypass both Nginx and the shim.

That path is deliberate, not an oversight. Nginx exists to accept network traffic, check the
API key, and route Anthropic-format requests to the shim. An agent on the same host needs
none of that: there's nothing to reverse-proxy, and it speaks the OpenAI API natively, so
there's nothing to normalize. Routing it through Nginx would add two process hops per
request for no benefit.

Two consequences worth stating explicitly:

- **Port 8001 must stay bound to `127.0.0.1`.** It has no authentication or encryption. When
  an agent depends on it, it's tempting to "just expose it" to a second machine. Don't:
  anything that can reach it gets unrestricted use of the model and sees every prompt.
  Remote clients should go through Nginx, which requires the API key.
- **A client that reads the context window from `/v1/models` adapts to the server.** Claude
  Code instead assumes ~200K tokens for a model name it doesn't recognise (unless
  `CLAUDE_CODE_MAX_CONTEXT_TOKENS` is set), and compacts to stay inside that. That
  assumption is why `--max-model-len` is 262144, as explained under
  [Throughput Expectations](#throughput-expectations-and-limitations).

**Availability.** `restart: "no"` is intentional (see the [risk table](#risk-summary)), so
that a crash stays visible instead of restart-looping silently. If a client can't tolerate
the model being down, give it a cloud fallback provider in its own configuration. Have it
fire only on connection failure or overload, so cloud billing stays a genuine fallback.

**Why the workload shape mattered.** Agentic traffic is dominated by structured tool calls
and file edits rather than free-form prose. That single fact selected the speculative
decoding method: DFlash wins on predictable output and loses slightly on novel text. A
prose-heavy deployment on identical hardware should make the opposite choice.

---

## The Platform: Why the DGX Spark Changes the Rules

Traditional GPU inference has a hard constraint: model weights must fit in VRAM, and VRAM is
a small, expensive, physically separate pool, typically 24–80 GB on a workstation. On those
machines a 35B-parameter model at full precision (BF16) needs ~70 GB and doesn't fit. Even
with aggressive quantization you're at the limit, and a KV cache large enough for 128K
context is out of the question.

The DGX Spark's GB10 Grace Blackwell changes this with unified memory. The CPU and GPU
share a single 128 GB pool rather than having separate ones. This isn't a software trick
like CPU offloading, which shuttles weights across PCIe. It's the hardware architecture:
the GPU's tensor cores address the entire 128 GB directly.

The advantage is **capacity, not raw bandwidth.** The unified pool runs at ~273 GB/s
(LPDDR5X), modest next to a discrete GPU's VRAM (an H100 moves >3 TB/s). What the Spark
buys you is the ability to hold a 35B model *and* a large KV cache in one pool, on one
device, without the multi-GPU rack that capacity would otherwise need. You trade peak
throughput for a workload that fits, which is why generation runs at ~110 tokens/second
(see [Throughput Expectations](#throughput-expectations-and-limitations)) rather than at
cloud-cluster speeds.

What this means in practice:

- **Room to spare:** a 35B model fits alongside a large KV cache.
- **KV pool:** ~2.45 M tokens, about **nine** concurrent conversations at the full
  262144-token context, or proportionally more at shorter contexts.
- **No copying:** CPU and GPU address the same physical pool, so there is no PCIe copy.

**NVFP4 quantization.** The `Qwen3.6-35B-A3B-NVFP4` checkpoint uses NVFP4, which GB10
supports. The main benefit is **memory**: 4-bit weights take ~22 GB instead of ~70 GB at
BF16, which is what lets the full 35B model fit alongside the large KV cache. On the compute
side, the current vLLM build runs these weights through the **Marlin weight-only path**.
Weights stay 4-bit in memory, so the size advantage is fully kept, but they are
dequantized to BF16 for each matrix multiply, which runs on standard BF16 tensor cores.

**This distinction was tested, and it doesn't matter here.** This checkpoint is
`W4A16` (4-bit weights, 16-bit activations), which can't use FP4 tensor cores on any GPU, so
vLLM's "your GPU does not have native support for FP4 computation" warning is misleading but
harmless. A `W4A4` checkpoint that *can* use them was ~16% slower overall, because it is
larger and this machine is memory-bound. See [TUNING.md §4.3](TUNING.md#43-bytes-per-token--why-the-nvidia-checkpoint).

> **Risk — Driver Version:** NVFP4 needs driver 580.x or later. Verify with `nvidia-smi`
> before investing time in anything else. Don't upgrade the driver on a DGX Spark; it is
> part of the platform image.

---

## The Model: Qwen3.6-35B and the MoE Tradeoff

The model, `Qwen3.6-35B-A3B-NVFP4`, reflects a specific engineering tradeoff between
capability and inference cost.

Mixture of Experts (MoE) is the key idea. A dense model with 35B parameters uses all 35B for
every token it generates. An MoE model splits its parameters into "experts" and routes each
token through only a few of them. Qwen3.6-35B has 35B parameters in total, but only ~3B are
active per forward pass: that is the "A3B" in the name, and its config routes each token
through 8 of 256 experts. This means:

- **Memory:** the footprint of a 35B model (~22 GB at NVFP4).
- **Compute:** the per-token cost of a ~3B model.
- **Knowledge:** the full 35B of learned knowledge is available, though not all at once.

**Why not a larger model?** Decode speed on this machine tracks the bytes read per token,
which means the *active* parameter count. A model with ~12B active parameters reads roughly
4× more per token and runs roughly 4× slower, whatever its total size. A large dense model
would also leave little memory for the KV cache. See
[TUNING.md §4.3](TUNING.md#43-bytes-per-token--why-the-nvidia-checkpoint).

**Why not a smaller model?** 7–8B models are fast, but they lack the multi-hop reasoning
coding agents rely on: following how a change in one file ripples through a codebase,
producing correct multi-file diffs, reasoning through dependency trees.

**On quantization quality:** NVIDIA produced this checkpoint with its own calibration
tooling. This project did not benchmark output quality against the BF16 original. It
validated functional behaviour instead (tool calling, reasoning, long-context recall). Check
quality on your own tasks.

### Using a Different Model

The stack isn't locked to Qwen3.6-35B; see [README → Swapping the model](README.md#swapping-the-model)
for the steps. The parts that are model-specific:

- **`--speculative-config`.** The DFlash draft model is trained for Qwen3.6 and won't work
  with anything else. Re-sweep `num_speculative_tokens` for any new model: the best value
  depends on the workload and hardware, and vendor defaults were badly wrong here.
- **`--reasoning-parser`, `--tool-call-parser`, `--language-model-only`.** These are
  Qwen3-specific; remove or adjust them.
- **Memory budget.** A model with heavier weights leaves less room for KV cache, which
  reduces achievable context length and concurrency.
- **Quantization.** NVFP4 checkpoints come with calibration baked in. Most open models are
  published as GPTQ, AWQ, or GGUF. vLLM supports GPTQ and AWQ natively.
- **`HF_HUB_OFFLINE=1`.** Point `--model` at a local directory under `/models`, not a hub ID.

---

## Repository Layout

```
dgxspark-vllm-qwen/
├── docker-compose.yml            ← the whole stack; every non-default flag is commented
├── .env.example                  ← copy to .env: LLM_API_KEY, MODEL_PATH, HF_TOKEN (optional)
├── download-models.sh            ← fetches the model + DFlash draft into MODEL_PATH
├── nginx/
│   ├── default.conf.template     ← front door: API-key check, routing, SSE settings
│   └── certs/                    ← optional TLS certs (gitignored)
├── shim/
│   └── shim.py                   ← Anthropic normalizing shim (stdlib only)
└── bench/                        ← validate.py, bench.py, sweep.py
```

## Key Configuration Choices

[`docker-compose.yml`](docker-compose.yml) is the source of truth. Each non-default flag has
its reasoning inline. The choices that most need explaining:

**`--language-model-only`** skips loading the vision encoder that ships in the Qwen3.6
checkpoint. A coding API never receives images, so loading it would only cost memory and
startup time.

**`--reasoning-parser qwen3` and `--tool-call-parser qwen3_coder`** enable structured output
parsing. Qwen3 wraps its chain of thought in `<think>...</think>` and uses its own tool-call
format. Without these parsers vLLM returns the raw tokens, and clients see malformed
responses. With them, reasoning arrives in a separate `reasoning` field (OpenAI API) or
`thinking` blocks (Anthropic API), and tool calls arrive as structured `tool_calls`.

**`--max-num-batched-tokens 8192`** raises the scheduler's per-step token budget from the
implicit default of 2048. vLLM warns explicitly that the default starves speculative
decoding of draft slots. To be honest, the measured gain sits **at the noise floor and is
not established**. The setting is kept because it is the documented-correct one for spec
decode, not because it was proven faster.

**`--safetensors-load-strategy prefetch`** loads weights in parallel with a prefetching
reader. It can be tuned further with `--safetensors-prefetch-num-threads` and
`--safetensors-prefetch-block-size`. It replaces vLLM 0.24's
`--load-format fastsafetensors`, which doesn't exist on 0.26+. If you're porting an older
recipe, that's the flag that changed name. It only affects loading; the benchmark showed no
throughput change.

**`--async-scheduling`** overlaps the scheduler step with model execution. **It isn't a
tuning knob here.** vLLM already enables it by default for this configuration, so adding it
changed nothing measurable.

It is kept explicit as a fail-loud guard:

- **With the flag set,** an incompatible configuration raises an error at startup.
- **Left unset,** vLLM silently disables it and logs only at `INFO`, which
  `VLLM_LOGGING_LEVEL=WARNING` hides.

So if a future drafter or vLLM release makes async scheduling incompatible, you get a loud
startup failure instead of a silent performance regression.

**`--speculative-config '{"method":"dflash","model":"/models/Qwen3.6-35B-A3B-DFlash","num_speculative_tokens":4}'`**
turns on speculative decoding with **DFlash**, a separate 0.77 GB six-layer dense draft
model. The draft model proposes 4 candidate tokens per step, and vLLM verifies all 4 in a
single batched forward pass of the full model. When candidates are accepted, the sequence
advances several tokens for the cost of one forward pass.

This is **the single largest tunable win in the stack: +44%** over no speculation (111.4 vs
77.3 tok/s at concurrency 1, on vLLM 0.26.1). Two non-obvious points, both established by
measurement in [TUNING.md](TUNING.md#41-speculative-decoding--the-lever-that-moved):

- **The draft length was swept, not copied.** The upstream recipe, for a different
  quantization of this model, uses 15. On this hardware 15 was the *worst* value tested, and
  at concurrency 4 it was **slower than disabling speculation entirely**. The number of
  accepted tokens barely changes with draft length: the draft model only ever gets 2–3
  tokens right, so everything beyond that is verification cost with no return.
- **DFlash was chosen over MTP because of the workload.** The checkpoint has built-in
  Multi-Token Prediction heads, which need no separate draft model. MTP is a perfectly good
  choice; it simply loses here on tool calls (−7.2%) and at concurrency 4 (−9.1%), while
  winning slightly on prose. Speculation pays off in proportion to how predictable the
  output is, and agentic traffic is mostly structured tool calls and code edits.
  - **A prose-heavy deployment should prefer MTP:**
    `'{"method":"mtp","num_speculative_tokens":3,"moe_backend":"triton"}'`.
  - **MTP's spec config *requires* `"moe_backend":"triton"`,** because its draft layer is
    unquantized BF16 and omitting the field crashes `EngineCore`.

**`depends_on: condition: service_healthy`.** The shim waits for vLLM to be healthy, and
Nginx waits for both. vLLM's `start_period` gives it time to load weights and compile
before failed healthchecks start counting.

> **Why 600s and not 300s.** Cold start was measured across ten container recreates at
> **341–461 seconds**. The old 300s `start_period` plus 5 retries × 30s gave 450s of grace,
> *less than an observed load time*. A slow start would flip the container to `unhealthy`,
> and because Nginx and the shim both wait for `service_healthy`, the whole stack would
> stall behind it. Set this from measured load time, not a guess.
>
> **Re-measured after enabling prefetch loading, and deliberately left at 600s.** With
> `--safetensors-load-strategy prefetch`, a recreate that reuses the `vllm-cache` volume
> now takes **285–294s** (n=2). That isn't grounds for lowering the budget. It's a
> warm-cache figure, and `start_period` must cover the cold case, which still pays ~117s of
> torch.compile and FlashInfer autotuning on top. Lowering a timeout to fit the fast path is
> exactly the failure this note exists to prevent.

**The `vllm-cache` volume** keeps vLLM's compiled artifacts (torch.compile graphs and
FlashInfer autotune results, ~285 MB) across container *recreates*. Without it they live in
the container's writable layer, which `docker compose down` destroys, so every recreate would
recompile from scratch.

**`MODEL_PATH`** comes from `.env` and defaults to `/data/models`. The `models` volume
bind-mounts that host directory into the container at `/models`, which is where `--model`
points. The directory must exist before the first `docker compose up`.

**`HF_HUB_OFFLINE=1` / `TRANSFORMERS_OFFLINE=1`.** Both models are on local disk, so nothing
needs to resolve from the Hugging Face hub at runtime. Offline mode removes a network
dependency from startup.

### A Note on FP8 KV Cache — and Why This Stack Doesn't Use It

This deserves an explicit explanation, because it's a tempting optimization that seems to
free a lot of memory, and the reason to avoid it isn't obvious.

The KV cache stores the key/value attention tensors for every token in every active context.
At BF16 (the default), the KV pool plus activations takes ~68.6 GiB in this stack.
Quantizing the KV cache to FP8 would roughly double its token capacity, which is a large gain.

The catch:

- **It needs calibration scales.** FP8 KV cache quantization needs per-tensor scales
  (`k_scale`, `v_scale`, `q_scale`) that tell vLLM how to rescale values before quantizing.
  They must be computed on representative data and baked into the checkpoint. **The
  `Qwen3.6-35B-A3B-NVFP4` checkpoint doesn't include them.**
- **Without them, values are silently clipped.** vLLM falls back to `scale=1.0`, meaning no
  rescaling. FP8 E4M3 can only represent values up to 448, so any attention value above 448
  is clipped without warning.
- **The resulting errors are the worst kind for coding.** On long contexts the model then
  computes attention over subtly wrong values. That produces wrong variable names,
  hallucinated API signatures that look plausible, or off-by-one logic that is hard to catch
  in review.

**Do not add `--kv-cache-dtype fp8` with this checkpoint.**

Two later findings refine this conclusion, but don't overturn it:

- **A different checkpoint does ship the scales.** The `unsloth/Qwen3.6-35B-A3B-NVFP4`
  variants include calibrated `k_scale`/`v_scale` tensors. With them, FP8 KV cache measured
  as safe (long-context recall passed) and cost **0% throughput**. So the objection above is
  specific to *this* checkpoint's missing calibration, not to FP8 KV cache as a technique.
- **It buys capacity, not speed.** Even where it works, FP8 KV changed throughput by nothing
  measurable. Adopt it only if KV capacity becomes the binding constraint. It isn't here:
  there's room for ~9 concurrent full-context sequences. That's why the faster NVIDIA
  checkpoint was kept over the unsloth one, despite the latter's working FP8 KV.

**Always verify before enabling.** Check that the checkpoint actually contains
`k_scale`/`v_scale` tensors rather than assuming it does, because the failure is silent.

### No pre-tuned MoE kernel config

vLLM can load a pre-tuned Triton config for its fused-MoE kernel, keyed by expert count,
GPU, and dtype. The config previously shipped here keyed `dtype=fp8_w8a8`, so it never
matched this NVFP4 model. Forcing it to match crashed Triton with `OutOfResources`, because
it was tuned for a GPU with ~3× GB10's shared memory per SM. It was removed. If you want a
tuned config, generate one with vLLM's `benchmark_moe.py` on this GPU and dtype, and check
that the `dtype=` and `device_name=` in its filename match before mounting it.

---

## How Clients Reach the Model

A common pattern in local LLM stacks is to put a translation proxy such as LiteLLM in front
of the inference server. Claude Code speaks the **Anthropic** API (`POST /v1/messages`),
while many inference servers historically spoke only the **OpenAI** API
(`POST /v1/chat/completions`), and the proxy converts one to the other.

**This stack needs a lighter layer.** vLLM serves `/v1/messages` natively. The one thing
that breaks is that Claude Code puts MCP instructions, skills, and IDE context in a
`role: "system"` message **inside** the `messages` array. vLLM's `/v1/messages` allows only
`user` and `assistant` roles there, and returns 400.

[`shim/shim.py`](shim/shim.py) is a dependency-free, stdlib-only Python proxy that does one
thing. It moves those system messages into the top-level `system` field and forwards the
request to vLLM. Everything else passes through untouched, including SSE streams, so the
Anthropic response shape survives, including the `thinking` blocks Claude Code expects.

```
Claude Code  (Anthropic /v1/messages) ─→  Nginx :80 /coding/v1/messages ─→ shim ─→ vLLM :8000
OpenAI clients (/v1/chat/completions) ─→  Nginx :80 /coding/v1/...      ─────────→ vLLM :8000
On-host agents (OpenAI API)           ─→  127.0.0.1:8001 ─────────────────────────→ vLLM :8000
```

> **Why not a full translation proxy like LiteLLM?** A general-purpose proxy translates
> *between* the Anthropic and OpenAI schemas. For a reasoning model that can be lossy: an
> Anthropic→OpenAI→Anthropic round-trip can drop `thinking` blocks, because the OpenAI schema
> represents reasoning differently. The shim fixes only the one thing that breaks and keeps
> the native response shape.

---

## Nginx: the Front Door

[`nginx/default.conf.template`](nginx/default.conf.template) is mounted into the stock
`nginx:alpine` image's `/etc/nginx/templates/` directory. At startup the image runs
`envsubst` over it, substituting `${LLM_API_KEY}` from `.env`, and writes the result to
`/etc/nginx/conf.d/default.conf`.

**Path-based routing.** Everything under `/coding/` goes to the coding model. The longer
prefix `/coding/v1/messages` (Claude Code) wins over `/coding/` and goes to the shim; every
other path goes straight to vLLM. Adding a second model later means a new location block,
with no change to existing client configurations.

**API key.** Every `/coding/` request must carry `LLM_API_KEY`, either as
`Authorization: Bearer <key>` or as `x-api-key: <key>`. Both forms are accepted because
clients differ:

- OpenAI SDKs, Continue, and Claude Code with `ANTHROPIC_AUTH_TOKEN` send `Authorization: Bearer`.
- Anthropic SDKs send `x-api-key`.

The check is in **both** location blocks. Checking only `/coding/` would leave Claude Code's
`/coding/v1/messages` route open. `docker-compose.yml` refuses to start if the key is empty,
because an empty key would match a request that simply sends no header.

**Upstream resolution.** The upstreams are set through variables plus
`resolver 127.0.0.11` (Docker's embedded DNS). That way nginx looks up `vllm-coding` and
`anthropic-shim` per request (cached 10s) rather than once at startup. With a static
`upstream` block, nginx keeps the container IP it saw at boot. Recreating `vllm-coding`, as
`bench/sweep.py` does, could then leave nginx proxying to a dead address until it was
restarted.

**Streaming.** vLLM streams responses as **Server-Sent Events over HTTP/1.1**, not
WebSockets. Two settings make tokens flow one at a time instead of arriving in bursts:

- `proxy_http_version 1.1` lets vLLM stream with chunked transfer encoding, instead of nginx
  falling back to HTTP/1.0, where the end of a response is signalled by closing the connection.
- `proxy_buffering off;` makes Nginx forward each chunk as it arrives rather than
  accumulating the response. Without it the stream still completes, but tokens land in clumps.

**Timeouts.** The 600s `proxy_read_timeout` and `proxy_send_timeout` apply *per read*.
They cap how long Nginx waits between chunks, not the total request duration. As long as a
token arrives at least every 600s the connection stays open, so very long generations
complete fine: at ~110 tokens/second a 128K-token generation takes ~20 minutes and never
trips the timeout. A 504 mid-stream means the model stalled for more than 600s on a single
read, for example during an unusually large prompt prefill. Raise the value if you hit it.

**`/health` and `/`** answer without a key and return fixed text, for liveness checks and a
quick "is it reachable" test. The nginx container's healthcheck uses `/health`.

**TLS.** The template ends with a commented HTTPS server block. To enable it:

1. Put `fullchain.pem` and `privkey.pem` in `nginx/certs/`.
2. Uncomment the block in the template and the `443:443` port in `docker-compose.yml`.
3. Recreate nginx.

Without TLS, the API key and all model I/O, including the source code in your prompts,
cross the network in plain text. On networks you don't fully control, use TLS or reach the
box over a VPN.

> **Risk — Docker bypasses ufw.** Ports published by Docker are inserted into iptables ahead
> of ufw's rules, so `ufw deny 80` won't stop LAN access to Nginx. The API key and your
> network boundary are the controls that apply. If only this machine needs access, publish
> `127.0.0.1:80:80` instead.

---

## Launch Sequence

```
[1] vllm-coding starts
       ↓  loads NVFP4 weights into unified memory (~22 GB)
       ↓  allocates the PagedAttention KV cache pool (~68.6 GiB, 2.45 M tokens)
       ↓  torch.compile, FlashInfer autotune, CUDA graph capture
       ↓  healthcheck passes at /health   ← 341–461s cold, 285–294s warm
[2] anthropic-shim starts (waits for vllm-coding healthy)
[3] nginx starts          (waits for vllm-coding + anthropic-shim healthy)
```

What vLLM is doing during startup:

1. **Allocating the KV cache.** PagedAttention divides the ~68.6 GiB pool into fixed-size
   blocks up front, so concurrent requests can be scheduled deterministically.
   `--gpu-memory-utilization 0.75` tells vLLM to commit 75% of the usable pool to weights
   and KV cache combined. That measured ~91.2 GiB, holding 2,453,340 KV tokens.
2. **Compiling.** torch.compile and FlashInfer autotuning pick kernels for this GPU. This is
   ~117s on a cold start, and it is cached in the `vllm-cache` volume.
3. **Capturing CUDA graphs** for common batch sizes, so decode steps replay a recorded graph
   instead of re-dispatching each kernel. A 294s start measured ~53s here.

> **Risk — Cold Start Time:** the stack looks unresponsive for several minutes while it
> starts. Nginx waits on vLLM's healthcheck, which is correct behaviour.

> **Risk — restart: "no":** containers don't restart automatically. If vLLM runs out of
> memory or hits a CUDA error, run `docker compose up -d` again yourself. Change to
> `restart: unless-stopped` if you want it back after reboots.

---

## How the Inference Optimizations Work Together

**Prefix caching** stores the computed KV tensors for prompt prefixes it has already seen.
Claude Code sends the same large system prompt on every request: several thousand tokens of
instructions about tools, file formats, and behaviour. Without prefix caching, every request
recomputes those tensors. With it, the first request pays the full cost and later requests
in the same session skip the shared prefix. For Claude Code's usage pattern (many requests
per session, identical system prompt), this is one of the highest-impact settings in the stack.

**DFlash speculative decoding** (see [Key Configuration Choices](#key-configuration-choices))
is the one lever that produces more output tokens per weight read. On a memory-bound box
that is exactly the constrained resource, which is also why kernel-level optimizations
measured as noise.

**PagedAttention** manages the KV cache as pages rather than pre-allocating a contiguous
block per sequence. That is what allows `--max-num-seqs 32` (32 requests in flight) without
each one reserving a full context's worth of KV cache up front. Pages are allocated as
context grows and freed as soon as a sequence completes.

---

## Port Reference

| Port | Service | Bound to | Auth | Use |
|---|---|---|---|---|
| `80` | Nginx | all interfaces | `LLM_API_KEY` | Claude Code, IDEs, SDKs, curl from the network |
| `443` | Nginx TLS | (disabled by default) | `LLM_API_KEY` | Same, encrypted. See [TLS](#nginx-the-front-door) |
| `8001` | vLLM direct | `127.0.0.1` | none | On-host agents, benchmarks, debugging |
| `8003` | anthropic-shim direct | `127.0.0.1` | none | Debugging the Anthropic path without Nginx |

---

## Throughput Expectations and Limitations

**Latency (time to first token).** For a request that hits the prefix cache (the second
request in a session with the same system prompt), first-token latency is typically 1–3
seconds. Cold requests take 3–8 seconds, depending on prompt length, because the whole
system prompt has to be recomputed.

**Throughput.** ~112 tok/s for a single request and ~235 tok/s aggregate at 4 concurrent
requests, with ~53% draft acceptance. A 500-token response streams in under 5 seconds.
Per-workload figures are in [README → Measured performance](README.md#measured-performance),
and the full sweep is in [TUNING.md](TUNING.md).

**The spread between workloads is larger than most configuration deltas.** Speculation pays
off in proportion to how predictable the output is, so code edits run ~20% faster than
novel prose on identical settings. Benchmark on prompts that resemble your actual traffic; a
single unrepresentative prompt can invert a conclusion.

**This is memory-bound, not compute-bound.** Roughly 0.9 GB of weights is read per forward
pass against a ~273 GB/s ceiling, putting the machine at about a quarter of peak bandwidth
without speculation. A sustained single-stream run showed 93% reported GPU utilization at
**26.7 W**: stalled on memory, not computing. That is why speculative decoding (more tokens
per weight read) is the dominant lever, and why kernel-level changes measured as noise.

**Context window.** The stack is configured for **256K tokens** (`--max-model-len 262144`),
deliberately more than Claude Code will use:

- **What Claude Code assumes.** For a model name it doesn't recognise, such as `coding`, it
  assumes a 200K window and compacts to stay inside it.
- **What went wrong at 131072.** Claude Code composed requests that vLLM rejected before
  generating anything (`prompt + max_tokens > max-model-len`). The user saw a spinner that
  never resolved.
- **The rule.** Setting the server's window *above* what the client assumes makes the
  client's own compaction fire safely below the real limit. If you set
  `CLAUDE_CODE_MAX_CONTEXT_TOKENS`, keep it well below 262144.
- **The cost is small.** Only 10 of the model's 40 layers use a growing KV cache
  (~20 KiB/token), so doubling the window cost about 2.7 GB per maximally long sequence,
  not twice the memory.

**Concurrency ceiling.** `--max-num-seqs 32` allows 32 requests in flight, but the real
ceiling is KV capacity. vLLM reports a pool of **2,453,340 tokens** and a
`kv_cache_max_concurrency` of **9.36**: about nine simultaneous sequences at the *full*
262144-token context, and proportionally more at realistic lengths. For a single user this
is never the bottleneck. For a small team, watch this metric rather than `--max-num-seqs`.

**The honest comparison to cloud.** A local stack trades latency predictability for cost and
privacy. Cloud APIs serve requests from large GPU clusters and can burst to much higher
throughput. This stack will vary more: a model reload or an unusually long context causes
noticeable pauses. In return, every token is generated on your own hardware.

---

## Risk Summary

| # | Risk | Severity | Mitigation |
|---|---|---|---|
| 1 | Driver < 580.x | Critical | Check `nvidia-smi` before building; NVFP4 requires it. Don't upgrade the driver on a DGX Spark |
| 2 | vLLM image < 0.26 | Critical | The compose flags (DFlash, `--safetensors-load-strategy`) need 0.26+. Build the pinned v0.30.0 image ([SETUP_GUIDE Phase 2](SETUP_GUIDE.md#phase-2--build-the-vllm-image)) |
| 3 | FP8 KV cache flag added | High | Do NOT add `--kv-cache-dtype fp8` with *this* checkpoint: it lacks `k_scale`/`v_scale`, so values are silently clipped. Verify the scales exist before enabling it on any checkpoint |
| 4 | API key leaked or weak | High | Generate it with `openssl rand -hex 32`; never commit `.env`; rotate by editing `.env` and recreating nginx |
| 5 | No TLS on Nginx | High | Plain HTTP on :80, so the key and prompts are readable on the wire. Enable TLS or use a VPN on untrusted networks |
| 6 | Docker bypasses ufw | Medium | Port 80 is reachable from the LAN regardless of ufw. Bind `127.0.0.1:80:80` if only this machine needs access |
| 7 | `restart: "no"` | Medium | Out-of-memory or CUDA errors need manual recovery. Kept so crashes stay visible; use `unless-stopped` or a client-side fallback if you need availability |
| 8 | Streaming timeouts | Medium | Raise `proxy_read_timeout` if you see 504s mid-stream |
| 9 | Cold start 6–8 min | Low | Measured 341–461s cold, 285–294s warm. Wait for the healthcheck; the stack is loading, not broken. `start_period` must exceed the slowest **cold** start |
