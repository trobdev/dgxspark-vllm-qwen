# Performance Tuning — What Drives Speed on GB10

This document holds the measurements behind every non-default setting in
[`docker-compose.yml`](docker-compose.yml): what was tested, how it was measured, what moved
decode speed and what didn't. It ends with a procedure for evaluating a new model or vLLM build.

- **Hardware:** GIGABYTE AI TOP ATOM: NVIDIA GB10 Grace Blackwell, 128 GB unified memory,
  ~273 GB/s memory bandwidth, driver 580.173.02.
- **Model:** `nvidia/Qwen3.6-35B-A3B-NVFP4`, served by vLLM.
- **Versions:** tuned on vLLM 0.26.1, re-validated at parity on 0.30.0 (§5).
- **Tools:** everything here was measured with the scripts in [`bench/`](bench/).

---

## 1. Summary

**Almost none of the hardware-level optimizations mattered. The one parameter copied from a
vendor recipe was actively harmful.**

| | starting config | tuned | change |
|---|---|---|---|
| decode @ concurrency 1 | 104.8 tok/s | **111.4 tok/s** | +6% |
| aggregate @ concurrency 4 | 174.3 tok/s | **227.8 tok/s** | **+31%** |
| speculative acceptance | 18.7% | **53.1%** | — |
| draft tokens burned per unit output | baseline | **−28%** | — |

*The starting config was DFlash speculative decoding at draft length 15, taken from an
upstream recipe.* Against no speculative decoding at all, the tuned config is **+44% at c=1**
and **+26% at c=4**.

**What produced this:**
- choosing a speculative decoder that matches how predictable the workload's output is;
- sweeping its draft length instead of trusting the published default.

**What produced nothing:**
- native FP4 tensor cores;
- the vendor-recommended MoE kernel backend and checkpoint;
- a "hardware-tuned" MoE kernel config;
- scheduler and load-path flags. Some of these were kept for other reasons (§4.4).

---

## 2. Why this machine behaves the way it does

Everything below follows from one fact: **interactive decode on a sparse MoE model is limited
by how many bytes must be read per token, not by how fast the GPU can multiply.**

**What the model reads per token:**
- **Experts:** 35B parameters in total, but only 8 of 256 experts per layer are active, about
  3B active parameters.
- **Attention:** it is a hybrid model. Only 10 of its 40 layers use a growing KV cache; the
  other 30 use linear attention with fixed state.
- **Size:** the weights are 22 GB on disk in NVFP4.
- **Per forward pass:** the active experts across 40 layers, the attention projections and the
  248,320-row LM head add up to roughly **1.6–1.9 B parameters, about 0.9 GB at NVFP4**.

Weights are read **once per target-model forward pass**, however many candidate tokens that
pass verifies. So bandwidth use tracks forward passes, not tokens/s:

| config | forward passes/s | implied GB/s | % of 273 GB/s |
|---|---|---|---|
| no speculative decoding | 77.3 | ~70 | **~25%** |
| DFlash @ 4 (deployed) | ~36 (111.4 tok/s ÷ ~3.1 tokens per pass) | ~32 | **~12%** |

**Speculation *lowers* bandwidth use.** It extracts ~3.1 tokens per weight read instead of 1.
That is exactly why it wins on this box: it buys more tokens per read rather than reading faster.

Three consequences:

1. **The machine is memory-bound at concurrency 1–8.** A sustained single-stream run showed
   93% reported GPU utilization while drawing only **26.7 W**: stalled SMs, not computation.
2. **Faster math does nothing.** FP4 tensor cores, better GEMM kernels and higher-TFLOP paths
   all optimize a resource that is already idle. This predicted, correctly, that every
   kernel-level experiment in §4.4 would be a wash.
3. **The ceiling is access efficiency, not raw bandwidth.** At ~12–25% of peak, the binding
   constraint is how efficiently memory is read:
   - scattered expert gathers;
   - skinny batch-1 matrix multiplies;
   - per-kernel launch overhead.

   The levers that work are reading fewer bytes per token, or producing more tokens per read.

**Physical limits** (cannot be changed with settings):

| Limit | Value | Consequence |
|---|---|---|
| Memory bandwidth | ~273 GB/s | Hard ceiling on weight streaming, ~300 forward passes/s for this model. Not currently binding (see above). |
| Memory capacity | 128 GB (~121 GiB usable) | Model + KV cache + every host process share one pool. |
| GPU count | 1× GB10 | `--tensor-parallel-size 1` is forced. |
| Compute capability | sm_121 | Upstream kernels (CUTLASS, FlashInfer, TRT-LLM) often target `family(100)` and exclude sm_12x; newer builds partly close this. |
| MoE gather pattern | 8 of 256 experts per layer | Scattered reads at batch 1 achieve far below peak bandwidth. Inherent to sparse MoE. |

---

## 3. How it was measured

### 3.1 Benchmark on representative prompts

[`bench/bench.py`](bench/bench.py) uses one prompt for each way an agentic coding workload
drives the model:

| workload | what it represents | output character |
|---|---|---|
| `tool_call` | agentic tool invocation, JSON arguments | highly structured, predictable |
| `code_edit` | modify a file, reproducing surrounding context | predictable |
| `prose` | long-form writing | novel, unpredictable |
| `reasoning` | general analysis and planning | mixed |

**The spread between workloads on one fixed config (~121 tok/s on code edits vs ~100 on
prose) is larger than most of the config deltas being measured.** A single unrepresentative
prompt can invert a conclusion. Two earlier benchmarks disagreed by 19 tok/s on identical
settings purely because of prompt content.

Even small wording changes move the numbers. Rewording just the persona sentence in the
`tool_call` system prompt shifted that workload from ~121 to ~148 tok/s (n=5, same server).
Speculative decoding rewards predictable output, and the wording changes what the model
writes. Treat benchmark prompts as fixed once you've published numbers measured with them.

### 3.2 Measure decode rate only

Requests are streamed. Time-to-first-token is recorded separately and excluded:

```
decode_tok_s = (completion_tokens - 1) / (t_last_token - t_first_token)
```

**How each result is computed:**
- **Prefill and prefix caching** don't touch the number, because time-to-first-token is excluded.
- **Warm-up first:** each config is warmed up before measuring, because the first run after a
  restart is 3–10× slower from JIT compilation.
- **Statistics:** the median per workload, then the median of medians as the c=1 headline.
- **Concurrency-4 aggregate:** total output tokens ÷ wall-clock time across 4 parallel streams.
- **Speculative acceptance:** the difference in vLLM's Prometheus counters across the run,
  `vllm:spec_decode_num_accepted_tokens_total` over `vllm:spec_decode_num_draft_tokens_total`.

### 3.3 Know your noise before declaring anything unmeasurable

**Precision of the rig.** With speculative decoding disabled, trial-to-trial spread is
**0.2–1.3%**, so the rig itself is precise to about 1%.

**Speculation adds its own variance.** With speculation on, acceptance-rate jitter adds
several percent. Six consecutive runs of one config spread about ±7% at c=1.

**Restarts add more.** The cleanest measurement of total noise came by accident: two n=14
runs of *functionally identical* configs, separated only by a container recreate. The flag
being "tested" (`--async-scheduling`) turned out to be vLLM's default already.

| run | c=1 median of medians | c=4 aggregate | acceptance |
|---|---|---|---|
| A | 115.9 tok/s | 229.1 tok/s | 53.6% |
| B | 118.6 tok/s | 245.4 tok/s | 54.6% |
| apparent delta | +2.3% | **+7.1%** | +1.0 pp |

Nothing changed between those runs. Had the default been missed, the honest-looking
conclusion would have been "async scheduling buys +7% at c=4", and it would have been wrong.

**Rules that follow:**

- **c=4 aggregate is the *noisier* metric,** despite summing four streams.
- **A container recreate is part of the experiment.** It re-runs CUDA graph capture and
  autotuning. A single before/after pair that needs a restart can't resolve <10% at c=4.
- **Treat <10% as *unresolved*, not *absent*.** Raise n and test the raw per-trial samples.
  The decisive DFlash-vs-MTP tool-call result (+7.2%, §4.2) would have been discarded as noise
  under a "±7% floor" rule. A Welch t-test at n=14 confirmed it.
- **How many trials:**
  - n=5 for exploratory sweeps;
  - **n≥14 plus a Welch t-test** for anything that decides a deployment;
  - re-measure the baseline in the same session.

### 3.4 Change one variable, against a matched control

[`bench/sweep.py`](bench/sweep.py) applies each config, then measures it:
1. rewrites `docker-compose.yml`;
2. force-recreates the container;
3. waits until it is healthy;
4. benchmarks it.

Because this is scripted, no manual step drifts between runs. Two rules came from getting
them wrong:

- **Benchmark tuned against tuned.** Comparing a tuned config against an untuned one produced
  a wrong conclusion twice.
- **Always run the matched control.** An early conclusion that the `flashinfer_b12x` backend
  "didn't help" was reached without running the same checkpoint *without* it.

---

## 4. Results

### 4.1 Speculative decoding — the lever that moved

Median tok/s across the four workloads (n=5 exploratory, n=14 for the finalists), vLLM 0.26.1:

| draft length | 2 | 3 | **4** | 6 | 8 | 15 | MTP 3 | **none** |
|---|---|---|---|---|---|---|---|---|
| **c=1** | 96.0 | 106.6 | **111.4** | 111.3 | 107.7 | 104.8 | 108.9 | **77.3** |
| **c=4** | 215.7 | 232.3 | **227.8** | 227.2 | 206.5 | 174.3 | 208.8 | **181.3** |
| **accept %** | 70.8 | 61.6 | 53.1 | 41.9 | 33.0 | 18.7 | 65.0 | — |

**(a) The vendor default was the worst setting tested.**
- **The setting:** draft length 15, from the upstream recipe for a different quantization of
  this model.
- **The result:** at c=4 it was **slower than disabling speculation entirely** (174.3 vs 181.3).
- **The diagnosis:** the *accepted* token count stays near ~11K whatever the draft length.
  The draft model only ever gets 2–3 tokens right, so everything past that is verification
  cost with no return.

**(b) The optimum is interior, and it moves with concurrency.** c=1 peaks at 4–6 and c=4 at
3. Sweep both regimes, not just the one you think you run in.

**(c) Acceptance rate is a diagnostic, not an objective.** Draft length 2 has the *best*
acceptance (70.8%) and the *worst* c=1 throughput. Short drafts are accepted reliably but
capture too little per verification pass.

**(d) Break statistical ties on efficiency.**
- **The tie:** draft lengths 4 and 6 are indistinguishable on speed (all |t| < 1.5 at n=14).
- **The tiebreak:** 4 reaches the same throughput at 53.1% vs 41.9% acceptance. That is 55K
  draft tokens instead of 76K for the same accepted output: same speed, 28% less wasted
  computation, lower variance.

### 4.2 Which speculative method — decided by output predictability

Two methods were viable:
- **MTP:** the multi-token-prediction head built into the checkpoint.
- **DFlash:** [`z-lab/Qwen3.6-35B-A3B-DFlash`](https://huggingface.co/z-lab/Qwen3.6-35B-A3B-DFlash),
  a separate 0.77 GB, 6-layer dense draft model.

DFlash @ 4 vs MTP @ 3, n=14, **means** (what the Welch t-test compares):

| workload | DFlash | MTP | delta | significant? |
|---|---|---|---|---|
| `tool_call` | 120.9 | 112.8 | **+7.2%** | yes |
| `code_edit` | 121.0 | 117.3 | **+3.2%** | yes |
| `prose` | 99.5 | 103.1 | −3.5% | yes |
| `reasoning` | 104.6 | 105.3 | −0.7% | no |
| **c=4 aggregate** | **227.8** | 208.8 | **+9.1%** | yes |

**DFlash wins exactly where output is predictable:**
- structured JSON tool calls;
- code edits that reproduce surrounding context.

It loses slightly on novel prose. Agentic coding traffic is dominated by tool calls and file
edits, so DFlash is the right trade here. **For a prose-heavy workload, MTP is the better
choice:**

```
--speculative-config '{"method":"mtp","num_speculative_tokens":3,"moe_backend":"triton"}'
```

**MTP requires `"moe_backend":"triton"`,** because its draft layer is unquantized BF16.
Omitting it crashes `EngineCore`. DFlash needs no such field.

**Draft-free (ngram) speculation loses badly:** 47.8 tok/s at 17.2% acceptance, against
123.7 for a real draft model. "It reads no extra weights, so it must win on a
bandwidth-limited box" is wrong. Verification cost scales with draft length however the draft
was produced, and a draft that is rarely accepted pays that cost for nothing.

### 4.3 Bytes per token — why the NVIDIA checkpoint

Holding everything else constant, decode speed tracked checkpoint size almost linearly:

| checkpoint | size | batch-1 tok/s |
|---|---|---|
| `nvidia/` W4A16 NVFP4 (deployed) | 22.0 GB | 116–119 |
| `unsloth/` `-Fast` W4A4 NVFP4 | 23.7 GB | 100–104 |
| `unsloth/` W4A4 NVFP4 | 26.5 GB | 95–99 |

**The W4A4 checkpoints can drive FP4 tensor cores, and they still lost.** W4A16 (4-bit
weights, 16-bit activations) *cannot* use FP4 tensor cores on any GPU, because FP4 tensor
cores compute FP4×FP4. It always runs the Marlin weight-only path. W4A4 can use them. But
this box isn't compute-bound, so the theoretically better quantization lost to the smaller
file: ~16% slower, measured across two independent harnesses.

**Two sub-levers when choosing a model:**

- **Active parameters.** A model with ~12B active parameters reads ~4× the bytes per token of
  a ~3B-active one, and runs roughly 4× slower whatever its total size. Total parameters only
  decide whether it fits.
- **Quantization width of the experts.** FP4 vs FP8 experts is 2× the bytes. Watch for
  "accuracy" variants that upcast some expert layers to FP8. They cost speed in proportion,
  which is why unsloth's non-Fast variant is slower than `-Fast`.

**Reading vendor benchmarks.** The unsloth figures (295 tok/s decode, 15,636 tok/s
throughput, "Marlin 2.5× slower") are correct, but for a different regime:

- **Their setup:** 1× B200 at 128 concurrent sequences.
- **Bandwidth:** B200 has ~8 TB/s, ~29× this machine.
- **Regime:** with 128 sequences, every weight read is shared across 128 tokens, so it is
  compute-bound. FP4 tensor cores and Marlin's dequantization overhead dominate there.

At concurrency 1–8 on 273 GB/s, both measured as a wash. **Before concluding a vendor is
wrong, check their concurrency and memory bandwidth.**

The unsloth checkpoints' one lasting advantage is calibrated `k_scale`/`v_scale` tensors,
which make FP8 KV cache safe (2× KV capacity). Revisit them only if KV capacity becomes the
binding constraint.

### 4.4 What didn't move decode speed

| experiment | result |
|---|---|
| Native FP4 vs Marlin, same W4A4 checkpoint | 104.1 vs 101.3 tok/s: **wash** |
| `--moe-backend flashinfer_b12x` vs auto-select, W4A4 `-Fast`, n=6 | 102.2 ± 2.8 vs 104.1 ± 1.7, Welch t = 1.26: **not significant**. Auto already picks a native FP4 path on sm_121 in vLLM 0.26.1. |
| FP8 KV cache (calibrated checkpoint) vs BF16 | **0% throughput change.** It buys capacity, not speed. |
| Pre-tuned MoE Triton config | **Never applied.** Its filename keyed `dtype=fp8_w8a8`, not NVFP4. Forcing a match crashed Triton with `OutOfResources` (shared memory 294912 required vs 101376 available), because it was tuned for a GPU with ~3× GB10's shared memory per SM. Removed: a future vLLM with looser matching would have loaded it and crashed. To tune MoE kernels, run vLLM's `benchmark_moe.py` on this GPU and dtype. |
| `--max-num-batched-tokens 8192` vs implicit 2048 | 101.3 → 108.1 (+7%), **at the noise floor, not established.** Kept because vLLM warns that 2048 starves speculative-decoding draft slots. |
| `--async-scheduling` | **Already vLLM's default** for this config, so adding it measured as noise (§3.3). Kept as a fail-loud guard (see the compose comment). |
| `--safetensors-load-strategy prefetch` | **Load path only.** Warm-cache recreate 341–461s → 285–294s (n=2), with no throughput change. |
| `--gpu-memory-utilization`, `--max-model-len` | Capacity only, no throughput effect. |

### 4.5 Concurrency — the multiplier for batchable work

Batching shares one weight read across many sequences:

```
c=1: 123.7      c=4: 277.1      c=8: 365.1   tok/s aggregate
```

*Measured with the MTP @ 3 config deployed at the time and an earlier prompt set, before the
four-workload harness. The absolute numbers aren't comparable with §4.1; the ratio is the point.*

That is ~3× aggregate at c=8, for ~2.6× worse per-stream latency. If work can be batched,
such as parallel agents or background jobs, this is the largest multiplier available. It is
also the regime vendors benchmark in (§4.3).

### 4.6 Problems found along the way

- **Context ceiling.**
  - **The problem:** Claude Code assumes a ~200K window for a model name it doesn't
    recognise. With `--max-model-len 131072`, requests were rejected before generation
    (`prompt + max_tokens > max-model-len`), and the user saw a spinner that never resolved.
  - **The fix:** raise the server's window *above* what the client assumes, to 262144.
  - **The cost** is small: only 10 of 40 layers keep a growing KV cache (~20 KiB/token).
- **Healthcheck too tight.**
  - **The problem:** `start_period` 300s plus 5 × 30s retries gave 450s of grace, but cold
    starts measured **341–461s**. A slow start would mark vLLM unhealthy and stall nginx and
    the shim, which both wait for it.
  - **The fix:** raised to 600s.
  - **Later:** after prefetch loading, warm starts dropped to 285–294s, and 600s was kept
    deliberately. `start_period` must cover the *cold* start.
- **Memory headroom.** Utilization was lowered from 0.85 to 0.75:
  - at 0.85 the whole box sat at ~122 GB of 128 GB;
  - at 0.75 vLLM commits ~91 GiB, and ~17 GiB of host memory stays free;
  - so the box survives days of uptime.

---

## 5. Version history: vLLM 0.30.0 re-validation

Image change only, with no flag changes:
- **vLLM:** `0.30.1.dev1+gbf1979fbd` (v0.30.0 plus the build repo's preset PRs).
- **FlashInfer:** 0.6.17 → 0.7.1.
- **Unchanged:** torch 2.11.0+cu130, CUDA 13.0, and the driver.

n=14, same harness and config:

| | 0.26.1 | 0.30.0 | change |
|---|---|---|---|
| `tool_call` | 120.9 | 115.8 | −4.2% |
| `code_edit` | 121.0 | 121.7 | +0.6% |
| `prose` | 99.5 | 99.0 | −0.5% |
| `reasoning` | 104.6 | 109.0 | +4.2% |
| **c=1 median of medians** | 111.4 | **112.4 tok/s** | +0.9% |
| **c=4 aggregate** | 227.8 | **234.8 tok/s** | +3.1% |
| acceptance | 53.1% | **53.0%** | — |

**Verdict: parity.** Every delta is inside the noise in §3.3; the −4.2% `tool_call` delta came
with a 20% within-run spread. Unchanged acceptance confirms DFlash behaves identically. A
serving-framework upgrade doesn't change bytes per token, so this is what §2 predicts.
`bench/validate.py` passed 7/7.

**Startup notes for 0.30.0:**

- **First start: 347s,** inside the cold range.
- **The autotune cache is per FlashInfer version,** in `flashinfer_autotune_cache/<version>/`.
  So the first start after an upgrade autotunes again (~26s here), even with a warm `vllm-cache`
  volume. Old version directories are safe to delete.
- **New warnings, all benign here:**
  - `Speculative decoding (method=dflash) is enabled but no KV cache group could be identified
    as the draft model's` reads like a misconfiguration, but acceptance is unchanged.
  - `Triton kernel JIT compilation during inference` is a one-time latency spike on the first
    requests after each start.
  - `python -m vllm.entrypoints.openai.api_server` is deprecated in favour of `vllm serve`.
    It still works; migrate before it is removed.
  - `Failed to import the DeepSelect extension` doesn't matter: this model doesn't use it.

---

## 6. Open levers

- **CUDA graph and compilation coverage.** At ~36 target forward passes/s across 40 layers,
  per-kernel launch overhead is worth profiling. It is the most plausible explanation for
  the gap between ~12% bandwidth use and the ceiling.
- **MoE gather efficiency.** The binding constraint is access pattern, not raw bandwidth
  (§2). Anything that improves expert-gather locality at batch 1 attacks it directly. No
  concrete lever has been identified yet.
- **Startup time.** In a 294s warm start, ~65s is FlashInfer autotuning *despite* cache hits
  and ~53s is CUDA graph capture. The weight loader was never the dominant cost.

---

## 7. Evaluating a new model or vLLM build

Run these in order. Steps 1–3 are cheap and rule out bad options before a 25 GB download.

**1. Read the checkpoint's quantization recipe.**
```bash
curl -sL https://huggingface.co/ORG/MODEL/resolve/main/config.json | python3 -c "
import json,sys; q=json.load(sys.stdin)['quantization_config']
print('quant_algo:', q.get('quant_algo'))
for g,v in q.get('config_groups',{}).items():
    w=v['weights']; a=v.get('input_activations') or {}
    print(g, f\"W{w['num_bits']}A{a.get('num_bits')}\", v.get('targets'))"
```

What the output means:
- **`W4A16`:** weight-only. Native FP4 is impossible on any GPU; Marlin is the only path.
- **`W4A4`:** native FP4 is available.
- **Expert layers upcast to FP8:** the checkpoint is larger, slower, and incompatible with
  `flashinfer_b12x`.

**2. Estimate bytes per token** before believing any benchmark:
`active_experts × 3 × hidden × moe_intermediate + shared + lm_head`, times bytes per
parameter. Compared with the current model, this predicts relative decode speed better than
any vendor claim.

**3. Check for `k_scale`/`v_scale` tensors** in `model.safetensors.index.json`:
- **Present:** `--kv-cache-dtype fp8` is safe.
- **Absent:** do not enable it. vLLM falls back to a scale of 1.0, and FP8 E4M3 silently
  clips any value above 448. That shows up as degraded long-context recall, not as an error.
  The deployed NVIDIA checkpoint has no scales.

**4. Check driver and CUDA compatibility before building.** **Don't update the GPU driver**
on a DGX Spark-class appliance. It is part of the platform image, and replacing it risks an
unbootable box. Adapt the container to the driver: the image's CUDA must be no newer than the
driver supports, or be covered by a `cuda-compat` layer.

**5. Protect the rollback image.** `build-and-copy.sh` reuses the target tag, so the previous
build is left untagged, one `docker image prune` away from deletion. First run
`docker tag vllm-node-v2:latest vllm-node:v2-previous`. Then stop vLLM before any
from-source compile: next to a resident model, an `-j 8` build triggered the OOM killer twice.

**6. Check what the new build does by default before crediting a flag.**
- **Defaults move:** `--async-scheduling` reads like a tuning knob, but vLLM 0.26+ already
  enables it whenever the config allows.
- **Flags get renamed:** 0.24's `--load-format fastsafetensors` doesn't exist on 0.26, which
  uses `--safetensors-load-strategy` instead.

A benchmark can't tell either case from "no effect". Read the resolution logic in vLLM's `config/`.

**7. Benchmark tuned against tuned, and sweep rather than copy.** `num_speculative_tokens`
and `--max-num-batched-tokens` depend on the hardware. Vendor defaults are tuned for vendor
hardware. Use n≥14 and a two-sample test for the final decision (§3.3).

**8. Validate correctness, not just speed.** Run `python3 bench/validate.py`. It covers
coherence, tool calling, `thinking` blocks through the shim, and long-context recall.

**9. Don't trust a hand-rolled smoke test on a reasoning model.**
- **The trap:** a quick `curl` with `max_tokens: 32` comes back with `content: null` and
  `finish_reason: length`. All 32 tokens went to thinking; Qwen3.6 spends ~220 reasoning
  tokens even on "say ok". It looks like a server-side reasoning limit, but it isn't.
- **What doesn't work:** `"reasoning": {"effort": "disabled"}` does nothing here (235
  reasoning tokens with it, 220 without). A `/no_think` system message doesn't either.
- **What works:** `"chat_template_kwargs": {"enable_thinking": false}`, which gives 0
  reasoning tokens.

---

## 8. What generalizes

For anyone tuning local inference on bandwidth-limited hardware:

1. **Establish which resource you're short of before optimizing.** One bandwidth calculation
   predicted that every kernel-level experiment here would be a wash. It was right.
2. **Bytes per token predicts decode speed.** Estimate active parameters × quantization width
   before downloading anything. Total parameters decide whether a model fits; active
   parameters decide how fast it runs.
3. **Never copy a vendor's speculative-decoding parameters.** Here the published default was
   worse than disabling the feature. Sweep it; the optimum is interior and shifts with concurrency.
4. **Match the speculative method to your output.** Predictable output (tool calls, code,
   structured formats) rewards aggressive drafting; novel prose doesn't.
5. **Benchmark on prompts that resemble your real traffic.** Here the spread between
   workloads exceeded almost every config delta.
6. **Measure your noise with a control before calling anything unmeasurable.** The apparent
   noise floor here was a property of the feature under test, not of the instrument.
7. **Read vendor benchmarks as claims about a specific regime.** Two correct benchmarks can
   disagree by 2.5× because they describe different machines.

**Realistic expectations on this hardware:**
- **This config:** ~112 tok/s single-stream, ~235 aggregate at c=4.
- **Spread:** expect ~20% between workload types; quote a range and say what it was measured on.
- **Heavier models:** a model with ~4× the active parameters runs ~4× slower, whatever the tuning.
- **Where the remaining wins are:** leaner checkpoints, better speculation, and batching work
  where possible. The first two are the same lever: fewer bytes read per token produced.
