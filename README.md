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

**Model:** `nvidia/Qwen3.6-35B-A3B-NVFP4` — 35B MoE (3B active), Blackwell NVFP4, 256K context, tool-calling,
with DFlash speculative decoding (~112 tok/s single-stream, ~235 tok/s at 4 concurrent —
see [Measured performance](#measured-performance)).
The stack is model-agnostic; see [Swapping the model](#swapping-the-model) to use a different one.

**Primary workload:** [Hermes Agent](OPTIMIZATION_REPORT.md), running on the host and
connecting **straight to `localhost:8001`** over the OpenAI API — it bypasses Nginx and the
shim, because it runs locally and speaks OpenAI natively, so there is nothing to proxy or
translate. Claude Code and other network clients come in through Nginx. The tuning in
[OPTIMIZATION_REPORT.md](OPTIMIZATION_REPORT.md) was driven by Hermes' traffic shape:
tool-call and code-edit heavy, which is what selected DFlash speculative decoding.

## Documentation

| Doc | Purpose |
|-----|---------|
| **README.md** (this file) | Quick start, API access, day-to-day operations |
| [SETUP_GUIDE.md](SETUP_GUIDE.md) | End-to-end setup from a fresh DGX Spark (Docker, NVIDIA toolkit, vLLM image build) |
| [IMPLEMENTATION_GUIDE.md](IMPLEMENTATION_GUIDE.md) | The "why" — architecture, design decisions, and the reasoning behind each config choice |
| [OPTIMIZATION_REPORT.md](OPTIMIZATION_REPORT.md) | What was tested, how it was measured, results, and why each setting was chosen |
| [PERFORMANCE_PLAYBOOK.md](PERFORMANCE_PLAYBOOK.md) | Ranked tuning levers, negative results, and how to evaluate a new model |
| [bench/](bench/) | Benchmark harness, config sweeper, and the functional validation suite |

## Measured performance

GIGABYTE AI TOP ATOM (GB10, 128 GB unified, driver 580.173.02), vLLM 0.30.0, the config in
`docker-compose.yml` as shipped. Measured 2026-10-04, n=14 per workload.

**Decode rate by workload** — single request (c=1), tokens/sec:

| Workload | tok/s | What it represents |
|---|---|---|
| `code_edit` | 121.7 | modify a file, reproduce surrounding context |
| `tool_call` | 115.8 | agentic tool invocation, JSON arguments |
| `reasoning` | 109.0 | general analysis |
| `prose` | 99.0 | novel, unpredictable text |
| **Headline (median of medians)** | **112.4** | |
| **4 concurrent requests, aggregate** | **234.8** | |

Speculative acceptance: **53.0%** (DFlash, 4 draft tokens).

**What the tuning bought** — same hardware, same model (vLLM 0.26.1, n=14 for finalists):

| Configuration | c=1 tok/s | c=4 aggregate |
|---|---|---|
| No speculative decoding | 77.3 | 181.3 |
| DFlash, 15 draft tokens (upstream recipe default) | 104.8 | 174.3 |
| MTP, 3 draft tokens | 108.9 | 208.8 |
| **DFlash, 4 draft tokens (deployed)** | **111.4** | **227.8** |

The upstream default draft length was *slower than no speculation at all* at c=4. Upgrading
vLLM 0.26.1 → 0.30.0 changed nothing measurable (every delta inside the noise floor).

**Other measured figures:**

| | |
|---|---|
| Long-context recall | 241,815-token prompt, specific line retrieved verbatim |
| Cold start (container start → serving) | 341–461s; 347s on the first 0.30.0 start |
| Warm recreate (`vllm-cache` volume populated) | 285–294s |
| GPU memory committed | ~91 GiB of ~121 GiB usable (`--gpu-memory-utilization 0.75`) |

**How these were measured:** decode rate only — streamed, with time-to-first-token excluded so
prefill and prefix-cache hits don't inflate the number; temperature 0; prompts drawn from a
real agent workload rather than a synthetic benchmark; medians over n trials. Run-to-run
noise is about ±2–3% at c=1 and ±7% at c=4, so smaller differences are not meaningful.
Workload choice matters more than most config changes — note the ~20% spread between `prose`
and `code_edit` on identical settings. Reproduce with `python3 bench/bench_hermes.py <label> 14`;
full method and history in [OPTIMIZATION_REPORT.md](OPTIMIZATION_REPORT.md).

## Prerequisites

- Docker Engine with the NVIDIA Container Toolkit installed and the runtime configured
- `vllm-node-v2:latest` image built locally (see SETUP_GUIDE.md — must be vLLM ≥ 0.19 for NVFP4)
- ~30 GB of free disk space for model weights (22 GB target model + 0.8 GB DFlash draft model)

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

Watch startup (a recreate with the `vllm-cache` volume warm measures 285–294s — weights
load, the KV cache pool is allocated, and CUDA graphs compile. A true cold start, with that
cache empty, adds roughly 117s of torch.compile and FlashInfer autotune on top. The
healthcheck allows 600s before it counts failures, which is sized for the cold case):

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
    base_url="http://<dgx-ip>/coding/v1",
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
| Update vLLM image       | See [Important: vLLM image requirement](#important-vllm-image-requirement) |

---

## GPU memory

With `--gpu-memory-utilization 0.75` vLLM commits ~91 GiB of the ~121 GiB usable pool,
leaving the rest as headroom:

| Allocation                | Approx size  |
|---------------------------|--------------|
| NVFP4 weights             | 21.9 GiB     |
| DFlash draft model        | 0.7 GiB      |
| KV cache + activations    | 68.6 GiB     |
| Free headroom             | ~30 GiB      |

Measured live at 93,381 MiB. vLLM reports a KV pool of 2,453,340 tokens — about
**9 concurrent sequences at the full 256K context**, proportionally more at shorter ones.

**Why 0.75 rather than higher:** this is unified memory, shared with the OS and every other
process, and host usage grows over days of uptime. At 0.85 the box sat at ~122 GB of 128 GB
with no margin. Utilization buys KV capacity, **not throughput**, so the headroom is free in
tokens/second terms.

If the model OOMs on startup, lower `--gpu-memory-utilization` further before reducing
`--max-model-len` — but see the note in
[IMPLEMENTATION_GUIDE.md](IMPLEMENTATION_GUIDE.md#throughput-expectations-and-limitations)
on why 262144 is set deliberately high for Claude Code.

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
3. Remove or replace `--speculative-config` — the DFlash draft model is specific to
   Qwen3.6 and will not work with another model
4. Drop `--reasoning-parser` / `--tool-call-parser` if the new model doesn't use Qwen3's formats
5. `docker compose up -d --force-recreate vllm-coding`
6. Re-sweep `num_speculative_tokens` if you add a draft model — see
   [PERFORMANCE_PLAYBOOK.md](PERFORMANCE_PLAYBOOK.md)

---

## Important: vLLM image requirement

NVFP4 quantization on the GB10 Blackwell architecture requires **vLLM ≥ 0.19**.
The `vllm-node-v2:latest` image must be rebuilt from `eugr/spark-vllm-docker` if your existing
build predates NVFP4 support. See [SETUP_GUIDE.md](SETUP_GUIDE.md) for rebuild instructions.
The deployed build is `0.30.1.dev1+gbf1979fbd` (vLLM v0.30.0 plus the build repo's preset PRs,
FlashInfer 0.7.1), validated 2026-10-04 — see
[OPTIMIZATION_REPORT.md §6.2](OPTIMIZATION_REPORT.md).

To update it:

```bash
docker tag vllm-node-v2:latest vllm-node:v2-previous   # keep a rollback — the build reuses the tag
docker compose stop vllm-coding                        # free ~91 GiB before a from-source compile
cd ~/spark-vllm-docker && git pull
./build-and-copy.sh -t vllm-node-v2 --vllm-ref <tag> --apply-preset-vllm-prs -j 8
cd ~/llm-stack && docker compose up -d && python3 bench/validate.py
```

`-t vllm-node-v2` is required: the script's default tag is `vllm-node`, which this compose
file does not use. Stopping vLLM first matters — a from-source build at `-j 8` next to a
resident model (~91 GiB committed) triggered the kernel OOM killer twice during the
2026-10-04 upgrade. With the model stopped, the same build completed with no OOM events
in 39m42s.
