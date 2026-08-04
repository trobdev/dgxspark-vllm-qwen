# Optimizing a 35B MoE for Agentic Inference on a 128 GB GB10 Box

**Subject:** `Qwen3.6-35B-A3B-NVFP4` served by vLLM, tuned for Hermes Agent as the primary
consumer, with secondary Claude Code use.
**Hardware:** GIGABYTE AI TOP ATOM — NVIDIA GB10 Grace Blackwell, 128 GB unified memory,
~273 GB/s memory bandwidth, driver 580.173.02.
**Date:** 2026-08-03. **Final commit:** `e73e4c7`.

---

## 1. Summary

Over a single extended session we tested seven categories of change against a local
35B-parameter mixture-of-experts model, measuring each one on prompts drawn from the actual
workload rather than synthetic benchmarks.

**The headline result is that almost none of the hardware-level optimizations mattered, and
the one parameter we had copied from a vendor recipe was actively harmful.**

| | before | after | change |
|---|---|---|---|
| decode @ concurrency 1 | 104.8 tok/s | **111.4 tok/s** | +6% |
| aggregate @ concurrency 4 | 174.3 tok/s | **227.8 tok/s** | **+31%** |
| speculative acceptance | 18.7% | **53.1%** | — |
| draft tokens burned per unit output | baseline | **−28%** | — |

*"Before" is the configuration actually running at the start of this session: DFlash
speculative decoding at draft length 15, taken from an upstream recipe.*

Against *no speculation at all*, the tuned configuration is **+44% at c=1** and **+26% at c=4**.

The changes that produced this were: choosing a speculative decoder matched to the
workload's output predictability, and sweeping its draft length instead of trusting the
published default. Changes that produced nothing: native FP4 tensor cores, the
vendor-recommended MoE kernel backend, the vendor-recommended checkpoint, and a
hardware-specific tuned kernel config.

---

## 2. Why this machine behaves the way it does

Everything below follows from one fact, so it is worth establishing first.

The model is a mixture-of-experts: 35B total parameters, but only 256 experts with 8 active
per token, giving roughly 3B *active* parameters. It is also a hybrid attention model — of
its 40 layers, only 10 do conventional growing-KV-cache attention (`full_attention_interval:
4`); the other 30 use linear attention with fixed state. Weights on disk are 22 GB in NVFP4.

Decoding one token requires reading every active weight from memory. A reasonable estimate of
what gets touched per token — active experts across 40 layers, attention projections, plus the
248,320-row LM head — is on the order of **1.6–1.9 B parameters, or roughly 0.9 GB at NVFP4**.

At the measured no-speculation rate of 77.3 tok/s, that is approximately **70 GB/s against a
~273 GB/s ceiling — around a quarter of peak.**

Two consequences drive the entire report:

1. **The machine is memory-bound, not compute-bound, at concurrency 1–8.** The arithmetic
   units spend most of their time waiting. A corroborating observation: a sustained
   single-stream run showed 93% reported GPU "utilization" while drawing only **26.7 W** —
   that is stalled SMs, not computation.
2. **Because we are not compute-bound, faster math does nothing.** Anything that makes
   multiplication faster — FP4 tensor cores, better kernels, higher-TFLOP paths — optimizes a
   resource that is already idle. This predicts, correctly, that every kernel-level experiment
   in §5 came out a wash.

The levers that *do* work on a memory-bound machine are: reduce bytes read per token, or get
more output tokens per weight-read. Speculative decoding is the second one, which is why it
dominates the results.

---

## 3. What we tested

| # | Category | Variants |
|---|---|---|
| 1 | Checkpoint / quantization | `nvidia/` W4A16_NVFP4 vs `unsloth/` W4A4 NVFP4 vs `unsloth/-Fast` |
| 2 | MoE kernel backend | Marlin (weight-only) vs `flashinfer_b12x` (native FP4) vs auto |
| 3 | KV cache dtype | bf16 vs fp8 (with calibrated scales) |
| 4 | Speculative decoding | none, ngram, MTP (2/3/5), DFlash (2/3/4/6/8/15) |
| 5 | Scheduler | `--max-num-batched-tokens` 2048 (implicit) vs 8192 |
| 6 | Context length | 131072 vs 262144 |
| 7 | Memory utilization | 0.85 vs 0.75 |
| 8 | Tuned MoE Triton config | mounted vs not |

---

## 4. Method

### 4.1 Benchmark on the real workload, not a synthetic one

This turned out to matter more than any single setting. Hermes drives the model in four
distinguishable ways, so the harness uses one prompt archetype for each:

| archetype | what it represents | output character |
|---|---|---|
| `tool_call` | agentic tool invocation, JSON | highly structured, predictable |
| `code_edit` | modify a file, reproduce surrounding context | predictable |
| `prose` | content development | novel, unpredictable |
| `reasoning` | general PA analysis | mixed |

**The per-workload spread on a single fixed configuration (121 tok/s on tool calls vs 100 on
prose) is larger than most of the configuration deltas we were trying to measure.** A single
unrepresentative prompt can therefore invert a conclusion. Two benchmarks earlier in this
project disagreed by 19 tok/s on identical settings purely because of prompt content.

### 4.2 Measure decode rate, not end-to-end latency

Requests are streamed via SSE. Time-to-first-token is recorded separately and excluded, so the
reported figure is pure decode throughput:

```
decode_tok_s = (completion_tokens - 1) / (t_last_token - t_first_token)
```

This removes prefill and prefix-cache effects from the number being compared. Each
configuration is warmed up before measurement — the first run after a container restart is
3–10x slower due to JIT compilation.

Reported statistics: **median** per workload (robust to the occasional stalled trial), then
median-of-medians as the c=1 headline. Concurrency-4 aggregate throughput is total output
tokens divided by wall-clock across 4 parallel streams, repeated 3–8 rounds.

Speculative acceptance is read from vLLM's Prometheus endpoint, differencing
`vllm:spec_decode_num_accepted_tokens_total` against `vllm:spec_decode_num_draft_tokens_total`
across the run.

### 4.3 Characterize the instrument before trusting it

We had previously recorded a "±7% noise floor" and were discarding sub-10% differences as
unmeasurable. **Running a no-speculation control corrected this.**

With speculation disabled, spread across trials collapsed to **0.2–1.3%** on every workload.
The measurement rig is precise to about 1%. The ±7% band is *acceptance-rate jitter intrinsic
to speculative decoding*, not instrument error.

This is a meaningful distinction: it means sub-10% differences are **unresolved, not absent**,
and are recoverable by raising n and applying a two-sample test. The decisive +7.2% tool-call
result in §5.1 would have been wrongly discarded under the old rule.

Accordingly: n=5 for exploratory sweeps, **n=14 plus a Welch t-test on raw per-trial samples**
for anything that decided a deployment.

### 4.4 Change one variable, with a matched control

Configurations are applied by rewriting `docker-compose.yml`, force-recreating the container,
polling until healthy, then benchmarking — scripted, so no manual step drifts between runs.
The original file is restored on exit.

Two rules learned by getting them wrong earlier in this project:

- **Benchmark tuned-vs-tuned.** Comparing a tuned configuration against an untuned one
  produced a wrong conclusion twice.
- **Always run the matched control.** We initially concluded the `flashinfer_b12x` backend
  "didn't help" without ever having run the same checkpoint *without* it.

---

## 5. Results

### 5.1 Speculative decoding — the only lever that moved

Median tok/s across the four workloads (n=5 exploratory, n=14 for finalists):

| draft len | 2 | 3 | **4** | 6 | 8 | 15 | mtp=3 | **none** |
|---|---|---|---|---|---|---|---|---|
| **c=1** | 96.0 | 106.6 | **111.4** | 111.3 | 107.7 | 104.8 | 108.9 | **77.3** |
| **c=4** | 215.7 | 232.3 | **227.8** | 227.2 | 206.5 | 174.3 | 208.8 | **181.3** |
| **accept%** | 70.8 | 61.6 | 53.1 | 41.9 | 33.0 | 18.7 | 65.0 | — |

Four findings:

**(a) The vendor default was the worst setting tested.** We had been running draft length 15,
taken from the upstream recipe for a different quantization of the same model. It was not
merely suboptimal — **at concurrency 4 it was slower than disabling speculation entirely**
(174.3 vs 181.3). The diagnostic: *accepted* token count stays near ~11K regardless of draft
length. The draft model only ever gets 2–3 tokens right; everything past that is verification
cost with no return.

**(b) The optimum is interior, and it moves with concurrency.** c=1 peaks at draft length 4–6;
c=4 peaks at 3. Sweeping only the regime you think you run in will mislead you.

**(c) Acceptance rate is a diagnostic, not an objective.** Draft length 2 has the *best*
acceptance (70.8%) and the *worst* c=1 throughput. Optimizing acceptance directly is a trap —
short drafts accept reliably but capture too little per verification pass.

**(d) Break statistical ties on efficiency.** Draft lengths 4 and 6 are indistinguishable on
speed (all |t| < 1.5 at n=14; 111.4 vs 111.3 and 227.8 vs 227.2). We chose **4** because it
reaches identical throughput at 53.1% vs 41.7% acceptance — 55K draft tokens instead of 76K
for the same accepted output. Same speed, 28% less wasted computation, lower variance.

### 5.2 Which speculative method — decided by output predictability

Two methods were viable: **MTP**, a multi-token-prediction head built into the checkpoint, and
**DFlash**, a separate 0.77 GB 6-layer dense draft model.

DFlash @ 4 vs MTP @ 3, n=14. Figures are **means**, since that is what the Welch t-test
compares; the medians in §5.1 differ slightly and are the right statistic for the headline
throughput numbers:

| workload | DFlash | MTP | delta | significant? |
|---|---|---|---|---|
| `tool_call` | 120.9 | 112.8 | **+7.2%** | yes |
| `code_edit` | 121.0 | 117.3 | **+3.2%** | yes |
| `prose` | 99.5 | 103.1 | −3.5% | yes |
| `reasoning` | 104.6 | 105.3 | −0.7% | no |
| **c=4 aggregate** | **227.8** | 208.8 | **+9.1%** | yes |

The pattern is mechanistically coherent rather than incidental: **DFlash wins precisely where
output is predictable** — structured JSON tool calls, code edits that reproduce surrounding
context — and loses slightly on novel prose. Speculation pays off in proportion to how
guessable the next tokens are.

Because agentic workloads are dominated by tool calls and file edits, this trade is favorable.
**Had the machine been used mainly for prose, MTP would have been the correct choice.** The
right answer is a property of the workload, not of the hardware.

A related negative result: **draft-free (ngram) speculation loses badly** — 47.8 tok/s at 17.2%
acceptance versus 123.7 for a real draft model. The intuition that a draft-free method should
win on bandwidth-limited hardware, because it reads zero extra weights, is wrong. Verification
cost scales with draft length regardless of how the draft was produced.

### 5.3 Everything at the kernel level was a wash

Exactly as §2 predicts for a memory-bound machine:

| experiment | result |
|---|---|
| Native FP4 (`flashinfer_b12x`) vs auto-select, same W4A4 checkpoint, n=6 | 102.2 vs 104.1 — **Welch t = 1.26, not significant** |
| Marlin vs native FP4, same checkpoint | 104.1 vs 101.3 — **wash** |
| FP8 KV cache vs bf16 | **0% throughput change** (it buys capacity, not speed) |
| Tuned MoE Triton config | **never applied** — dtype mismatch; forcing a match crashed Triton with `OutOfResources` |
| `--max-num-batched-tokens` 8192 | +7%, **at the noise floor — not established.** Kept anyway; vLLM warns the implicit 2048 starves draft slots |

### 5.4 The vendor checkpoint we did not adopt

A well-regarded community checkpoint publishes a genuine W4A4 NVFP4 quantization — 4-bit
weights *and* 4-bit activations — which can drive FP4 tensor cores, unlike the NVIDIA
checkpoint's W4A16 (weight-only, 4-bit weights with 16-bit activations, which mathematically
*cannot* use FP4 tensor cores on any GPU, including a B200).

We tested it thoroughly, tuned against tuned. **It ran ~16% slower on this hardware**
(measured across two independent harnesses at different times), for a simple reason: the
checkpoint is larger. Decode speed tracked checkpoint bytes almost linearly — 22 GB beat
23 GB beat 25 GB, roughly in proportion. On a bandwidth-bound machine, size *is* speed, and
the theoretically superior quantization lost to the smaller file.

**This is not a case of a vendor being wrong.** Their published benchmarks are 1× B200 at 128
concurrency: ~8 TB/s of memory bandwidth, roughly 29× this machine, with weight reads
amortized across many sequences. That regime is **compute-bound**, where FP4 tensor cores and
kernel dequantization overhead matter enormously. Both sets of numbers are correct; they
describe different machines running different workloads.

The checkpoint's one durable advantage is a working FP8 KV cache (it ships calibrated
`k_scale`/`v_scale` tensors), giving 2× KV capacity at zero throughput cost. That is worth
revisiting only if KV capacity becomes the binding constraint — at 0.75 utilization with
~17 GiB spare, it is not.

### 5.5 Fixes found along the way

Not performance work, but found by the same process:

- **Context ceiling.** Claude Code assumes a ~200K context window for any custom endpoint and
  cannot be told otherwise. With `--max-model-len 131072`, requests were hard-rejected before
  generation with `prompt + max_tokens > max-model-len`, surfacing as a spinner that never
  resolved. Fixed by raising the server's window *above* what the client assumes.
- **Healthcheck too tight.** `start_period` was 300s with 5×30s of retry grace = 450s total,
  but measured cold start across 10 container recreates is **341–461s** — below an observed
  load time. A slow start would flip the container unhealthy and stall nginx and the API shim,
  both of which gate on `service_healthy`. Raised to 600s.
- **Latent kernel hazard.** The mounted MoE tuning config was for a different dtype and never
  matched. Harmless today, but a future vLLM loosening its matching rules would have crashed
  the server. Unmounted.
- **Memory headroom.** Utilization 0.85 → 0.75, taking committed GPU memory from ~122 GB to
  ~91 GB and leaving ~17 GiB free, so the box survives days of uptime without pressure.

---

## 6. Final configuration

```yaml
--model                     /models/Qwen3.6-35B-A3B-NVFP4
--max-model-len             262144
--gpu-memory-utilization    0.75
--max-num-seqs              32
--max-num-batched-tokens    8192
--enable-prefix-caching
--language-model-only
--reasoning-parser          qwen3
--enable-auto-tool-choice
--tool-call-parser          qwen3_coder
--speculative-config '{"method":"dflash",
                       "model":"/models/Qwen3.6-35B-A3B-DFlash",
                       "num_speculative_tokens":4}'
```

**Measured:** 110–111 tok/s at c=1, ~227 tok/s aggregate at c=4, 53% speculative acceptance,
93,381 MiB GPU (~91 GiB), ~17 GiB host memory free.

**Validated 7/7** before commit: OpenAI tool calling, tool-result round-trip, reasoning/content
separation, **241,815-token context recall** (retrieved a specific line verbatim), Anthropic
`/v1/messages` through nginx and the normalizing shim with `thinking` blocks preserved, SSE
streaming, and model metadata.

---

## 7. What generalizes

For anyone tuning local inference on bandwidth-limited hardware:

1. **Establish which resource you are actually short of before optimizing.** One bandwidth
   calculation predicted, in advance, that every kernel-level experiment here would be a wash.
   It was right, and it would have saved most of the session.
2. **Bytes per token is the dominant predictor of decode speed.** Estimate active parameters ×
   quantization width before downloading anything. Total parameter count only decides whether
   the model fits; *active* parameters decide how fast it runs.
3. **Never copy a vendor's speculative-decoding parameters.** This is the one setting where a
   published default was not just suboptimal but worse than disabling the feature. Sweep it —
   the optimum is interior and shifts with concurrency.
4. **Match the speculative method to your output distribution.** Predictable output (tool
   calls, code, structured formats) rewards aggressive drafting; novel prose does not. The
   same hardware justifies different answers for different workloads.
5. **Benchmark on prompts that resemble your real work.** Here, the spread between workload
   types on a fixed configuration exceeded almost every configuration delta we measured.
6. **Characterize your measurement noise with a control before declaring anything unmeasurable.**
   Our "noise floor" turned out to be a property of the feature under test, not the instrument
   — and treating it as a floor would have discarded the session's decisive result.
7. **Read vendor benchmarks as regime-specific claims.** Check concurrency and memory bandwidth
   before concluding anyone is wrong. Two correct benchmarks can disagree by 2.5× because they
   describe different machines.

---

## 8. Reproducing this

The benchmark harness, sweep driver, and validation suite are small standalone Python scripts
using only the standard library:

- **`bench_hermes.py`** — four workload archetypes, streamed, decode-rate isolated, medians
  over n trials, plus concurrency-4 aggregate and acceptance metrics.
- **`sweep.py`** — rewrites the compose file, force-recreates, waits for health, benchmarks,
  restores on exit. Sweeps are declarative lists of configs.
- **`validate.py`** — seven functional checks across both API surfaces, run before any commit.

Companion documents in this repository: **`PERFORMANCE_PLAYBOOK.md`** (ranked levers,
negative results, upgrade evaluation procedure) and **`V2_PLAN.md`** (rollback procedure and
risk register).

---
