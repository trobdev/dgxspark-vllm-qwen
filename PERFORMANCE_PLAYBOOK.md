# Performance Playbook — GB10 / DGX Spark-class Box

Derived from measured results on 2026-08-03 (GIGABYTE AI TOP ATOM, GB10, 128 GB unified,
driver 580.173.02, vLLM 0.26.1rc1). Written to be reused for future model upgrades
(Qwen3.8, etc.) rather than re-derived each time.

---

## 0. The one-paragraph model of this machine

Interactive single-stream decode on a sparse MoE is dominated by **how many bytes must be
read to produce one token**, not by how fast the GPU can multiply. Your GB10 has ~273 GB/s
of unified memory bandwidth and FP4 tensor cores. The FP4 tensor cores are largely
irrelevant to you, because you are never waiting on math. **But you are also not at the
bandwidth ceiling** — measured effective throughput is ~11% of theoretical, so the binding
constraint at batch-1 is memory *access efficiency* (scattered MoE expert gathers, skinny
batch-1 GEMMs, per-layer kernel overhead), not raw bandwidth.

---

## 1. Physical limits — cannot be changed

| Limit | Value | Consequence |
|---|---|---|
| Unified memory bandwidth | ~273 GB/s | Hard ceiling on weight streaming. Not currently binding (we use ~11%). |
| Unified memory capacity | 128 GB (121 GiB usable) | Caps model size + KV cache + host processes combined. CPU and GPU share one pool. |
| Single GPU | 1x GB10 | No tensor parallelism without a second box. `--tensor-parallel-size 1` is forced. |
| Compute capability | sm_121 | Consumer/workstation Blackwell. Upstream kernel coverage (CUTLASS/FlashInfer/TRT-LLM) frequently gates on `family(100)` and excludes sm_12x. Partially mitigated by newer builds — see §5. |
| MoE gather pattern | 8 of 256 experts/layer | Scattered reads at batch-1 achieve far below peak bandwidth. Inherent to sparse MoE at low batch. |

**Do not** try to fix these with settings. Buying performance here means different hardware
or a second Spark.

---

## 2. What actually drives decode speed — ranked by measured leverage

### Tier 1 — Bytes per token (dominant, roughly linear)

`bytes/token ≈ active_params × bytes_per_param`

This is the single biggest lever and the one most under your control at model-selection
time. Measured, holding everything else constant:

| Checkpoint | Size | batch-1 tok/s |
|---|---|---|
| nvidia W4A16 | 22.0 GB | 116-119 |
| unsloth `-Fast` W4A4 | 23.7 GB | 100-104 |
| unsloth non-Fast W4A4 | 26.5 GB | 95-99 |

Speed tracked size almost proportionally. Two sub-levers:

- **Active parameters.** A3B (~3B active) vs A12B (~12B active) is ~4x the bytes/token.
  A 120B-A12B model will be roughly 1/4 the speed of a 35B-A3B, regardless of total size.
  **Active params matter; total params only matter for fitting in memory.**
- **Quantization width.** FP4 vs FP8 on the *expert* weights is 2x. Watch for vendor
  "accuracy" variants that upcast some expert layers to FP8 — they cost speed
  proportionally (this is exactly why unsloth non-Fast is slower than `-Fast`).

### Tier 2 — Tokens per forward pass (speculative decoding)

Speculation multiplies output per weight-read. This is the main *algorithmic* lever on a
memory-limited box, and the **single largest tunable win available: +45%**.

Definitive sweep (2026-08-03) on Hermes-representative prompts — tool calls, code edits,
prose, reasoning — median tok/s, n=5 (n=14 for the finalists):

| draft len | 2 | 3 | **4** | 6 | 8 | 15 | mtp=3 | **none** |
|---|---|---|---|---|---|---|---|---|
| c=1 | 96.0 | 106.6 | **111.4** | 111.3 | 107.7 | 104.8 | 108.9 | **77.3** |
| c=4 | 215.7 | 232.3 | **227.8** | 227.2 | 206.5 | 174.3 | 208.8 | **181.3** |
| accept% | 70.8 | 61.6 | 53.1 | 41.9 | 33.0 | 18.7 | 65.0 | — |

**Deployed: DFlash @ 4.** Four rules fall out of this table:

1. **Do not copy the vendor default.** 15 (upstream's FP8 recipe) was the *worst* setting
   tested and was **slower than no speculation at all at c=4** (174.3 vs 181.3). The
   accepted-token count is roughly constant (~11K) at every draft length — everything past
   ~3 tokens is pure verification waste.
2. **There is an interior optimum, and it moves with concurrency.** c=1 peaks at 4-6; c=4
   peaks at 3. Sweep both regimes, not just the one you think you run in.
3. **Acceptance rate is a diagnostic, not an objective.** Draft length 2 has the *best*
   acceptance (70.8%) and the *worst* c=1 throughput. Optimising acceptance is a trap.
4. **Break ties on efficiency, not the point estimate.** 4 and 6 are statistically
   identical (all |t| < 1.5, n=14). 4 wins on 53.1% vs 41.7% acceptance — same speed,
   28% fewer draft tokens burned, lower variance.

**DFlash vs MTP.** DFlash (`z-lab/Qwen3.6-35B-A3B-DFlash`, a separate 0.77 GB 6-layer
dense draft model) beats MTP on structured output and loses slightly on free text —
exactly the shape you'd predict, since speculation pays off when output is predictable:

| workload | dflash4 | mtp3 | delta | Welch |
|---|---|---|---|---|
| tool_call | 120.9 | 112.8 | **+7.2%** | significant |
| code_edit | 121.0 | 117.3 | **+3.2%** | significant |
| prose | 99.5 | 103.1 | -3.5% | significant |
| reasoning | 104.6 | 105.3 | -0.7% | not significant |
| c=4 agg | 227.8 | 208.8 | **+9.1%** | significant |

Agentic workloads are tool-call dominated, so this trade is favourable. **Pick the spec
method to match your output distribution** — if the box were mainly writing prose, MTP
would be the right choice.

Structural notes: with `method: mtp` the draft layer is **unquantized BF16**, which is why
its spec config must carry `"moe_backend":"triton"` — omitting it crashes EngineCore.
DFlash needs no such flag. Draft-free (ngram) speculation **loses badly** (47.8 tok/s,
17.2% acceptance): verification cost scales with draft length regardless of how the draft
was produced, so reading "zero extra weights" does not save you.

### Tier 3 — Amortization (concurrency)

Batching reuses one weight-read across many tokens. Measured aggregate throughput:

```
c=1: 123.7      c=4: 277.1      c=8: 365.1   tok/s
```

~3x aggregate at c=8 for ~2.6x worse per-stream latency. **If work can be batched
(parallel agents, background jobs), this is the largest available multiplier.** It is also
the regime vendors benchmark in — which is why their numbers don't match yours (§6).

### Tier 4 — Avoiding work entirely

- `--enable-prefix-caching` — large win for Claude Code, whose long system prompt/context
  repeats every turn. Already enabled.
- KV cache dtype — capacity, **not** speed. FP8 KV measured 0% throughput change. It buys
  2x KV capacity, which converts to *headroom* (run at lower `--gpu-memory-utilization`)
  or *more concurrent long sessions*.

### Tier 5 — Kernel/scheduler tuning (small but real)

- `--max-num-batched-tokens 8192`: gain **NOT established** — the observed 101.3 -> 108.1
  sits exactly at the +-7% noise floor (see S8). Keep it anyway: vLLM explicitly warns the
  implicit 2048 starves spec-decode draft slots, so it is the documented-correct setting.
- MoE backend choice: **no measurable difference**, tested properly on the checkpoint
  unsloth's guidance targets (`-Fast`, W4A4), backend as the only variable, n=6:

  | `-Fast` @ c=1 | mean | sd |
  |---|---|---|
  | `--moe-backend flashinfer_b12x` | 102.2 | 2.8 |
  | auto-select | 104.1 | 1.7 |

  +1.8% favouring auto, Welch t=1.26 — not significant; flips to -1.4% at c=4.
  **Critically, auto also emits zero Marlin warnings** — vLLM 0.26.1 already selects a
  native FP4 path for a W4A4 checkpoint on sm_121, so the explicit flag is redundant on
  this build rather than wrong. unsloth's "mandatory on DGX Spark" guidance was correct
  for older vLLM where sm_121 auto-selection failed. **Keep the flag anyway** — it costs
  nothing and pins behaviour across future vLLM upgrades.

---

## 3. What does NOT help on this platform (measured, not assumed)

| Thing | Verdict |
|---|---|
| Native FP4 tensor cores (`flashinfer_b12x`) | **No gain** at c=1-8. Wash vs Marlin on identical checkpoints (104.1 vs 101.3). You aren't compute-bound. |
| Chasing higher-TFLOP kernels generally | Same reason. |
| Larger `--max-model-len` | Free. Costs only KV pool allocation, not speed. |
| `--gpu-memory-utilization` | Affects capacity/headroom, not throughput. |

---

## 4. Untested levers, highest-expected-value first

These are the concrete open opportunities as of 2026-08-03:

1. **The MoE tuning config has never loaded.** vLLM logs on every start:
   `Using default MoE config. Performance might be sub-optimal! Config file not found at
   /moe-configs/E=256,N=512,device_name=NVIDIA_GB10.json`. The mounted file has extra
   `dtype=`/`block_shape=` suffixes and does not match the expected name. Given the ~11%
   bandwidth efficiency, kernel tile selection is a plausible contributor. **Cheapest
   untested lever.**
2. **Draft-free speculation.** vLLM 0.26.1 offers `ngram`, `ngram_gpu`, and `suffix`.
   These read **no extra weights**, unlike MTP's BF16 draft layer — structurally
   attractive on a bandwidth-limited box, and coding workloads are highly repetitive
   (ideal for n-gram/suffix matching). Note: ngram crashed on vLLM 0.22.1rc1; that was
   four versions ago and is worth retrying.
3. **`qwen3_5_mtp` instead of generic `mtp`.** A model-family-specific MTP implementation
   exists for this architecture; we use the generic path.
4. **`dflash`.** Upstream ships `recipes/qwen3.6-35b-a3b-fp8-dflash.yaml`, so it is
   considered viable for this model family.
5. **CUDA graph / compilation coverage.** At ~36 forward passes/sec across 40 layers,
   per-kernel launch overhead is worth profiling before assuming it's negligible.

---

## 5. Upgrade playbook — evaluating a new model or vLLM build

Run in this order. Steps 1-3 are cheap and disqualify bad options fast.

**1. Read the checkpoint's quantization recipe before downloading 25 GB.**
```bash
curl -sL https://huggingface.co/ORG/MODEL/resolve/main/config.json | python3 -c "
import json,sys; q=json.load(sys.stdin)['quantization_config']
print('quant_algo:', q.get('quant_algo'))
for g,v in q.get('config_groups',{}).items():
    w=v['weights']; a=v.get('input_activations') or {}
    print(g, f\"W{w['num_bits']}A{a.get('num_bits')}\", v.get('targets'))"
```
- `W4A16` -> weight-only; native FP4 impossible **on any GPU**; Marlin is the only path.
- `W4A4` -> native FP4 available.
- **Check whether expert layers are upcast to FP8.** If yes: larger, slower, and
  incompatible with `flashinfer_b12x`.

**2. Estimate bytes/token before believing any benchmark.**
`active_experts × 3 × hidden × moe_intermediate + shared + lm_head`, times bytes/param.
Compare against your current model. **This predicts relative decode speed better than any
vendor claim.**

**3. Check for `k_scale`/`v_scale` tensors** in `model.safetensors.index.json`. Present ->
`--kv-cache-dtype fp8` is safe. Absent -> it will silently clip; do not enable.

**4. Verify driver/CUDA compatibility BEFORE building.** Never update the GPU driver
(see the hard constraint in `V2_PLAN.md`). Confirm the target CUDA base image is <= what
the driver natively supports, or that a `cuda-compat` layer covers it.

**5. Benchmark tuned-vs-tuned.** Comparing a tuned config against an untuned one produced
a wrong conclusion twice in one session. Fix the tuning first, then compare.

**6. Sweep, don't copy.** `num_speculative_tokens` and `--max-num-batched-tokens` are
hardware-dependent. Vendor defaults are tuned for vendor hardware.

**7. Validate correctness, not just speed.** Coherence, tool calling, thinking blocks
through the shim, and long-context recall (FP8 KV clipping shows up as degraded recall,
not as an error).

---

## 6. How to read vendor benchmarks

Vendor numbers are usually **not wrong** — they are measured in a different regime.

unsloth's Qwen3.6 NVFP4 figures (295 tok/s decode, 15,636 tok/s throughput, "Marlin 2.5x
slower") are **1x B200 @ 128 concurrency**. B200 has ~8 TB/s HBM3e — roughly **29x** your
bandwidth — and at 128 concurrent sequences each weight read amortizes across 128 tokens,
putting them firmly in a **compute-bound** regime where FP4 tensor cores and Marlin's
dequant overhead dominate. At concurrency 1-8 on 273 GB/s you are in a completely
different regime, where those factors measured as a wash.

**Before concluding a vendor is wrong, check their concurrency and memory bandwidth.**
Both results can be correct simultaneously.

---

## 7. Realistic expectations

- Current: **~117-124 tok/s** batch-1, ~365 tok/s aggregate at c=8, on a 22 GB / 3B-active
  FP4 MoE.
- Theoretical bandwidth ceiling: ~330 forward passes/sec; we achieve ~36. The gap is MoE
  gather inefficiency + batch-1 GEMM shape + kernel overhead. **Some is recoverable
  (§4), much is inherent to sparse MoE at batch-1.**
- A larger-active-param model (A12B) will be ~4x slower per token regardless of tuning.
- The biggest realistic wins available: smaller/leaner checkpoints, better speculation,
  and batching work when possible.


---

## 8. Measurement noise — read before trusting any delta

Six consecutive runs of the **identical** config at c=1:

```
116.3  123.8  120.8  125.1  123.3  122.1    mean 121.9, spread ~ +-7%
c=4:  278.9  286.5  283.6                   spread ~ +-2.7%
```

**Anything under ~10% at c=1 is not distinguishable with 3 trials.** Consequences for the
results in this document:

| Comparison | Delta | Verdict |
|---|---|---|
| MTP 3 vs 2 | +20% | Real |
| MTP 3 vs 5 | +13% | Real |
| ngram vs MTP=3 | -60% | Real |
| NVIDIA vs unsloth -Fast | +14-15% | Real (both benchmarks agree) |
| Marlin vs flashinfer_b12x | +3% | Noise |
| qwen3_5_mtp vs mtp | +-8% | Noise |
| --max-num-batched-tokens 8192 | +7% | **At the noise floor — not established** |

**Method:** warm up first (the first run after a restart is 3-10x slower from JIT), run at
least 5 trials for anything you intend to act on, and re-measure the baseline in the same
session rather than comparing against a number from hours earlier.

### 8.1 The noise floor is not instrument error (important correction)

Running the **no-speculation** control produced spreads of **0.2-1.3%** across every
workload. The measurement rig is therefore precise to ~1%; the +-7% band above is
**acceptance-rate jitter intrinsic to speculative decoding**, not a limit on what can be
resolved.

Consequence: differences under 10% are *not* automatically unknowable — they are
recoverable by raising n and testing the raw per-trial samples rather than eyeballing
medians. The dflash-vs-mtp comparison (+7.2% on tool calls) was confirmed significant at
n=14 by a Welch t-test, and would have been wrongly dismissed as noise under the old rule.

**Revised rule:** treat <10% as *unresolved*, not *absent*. If the decision matters, run
n>=14 and a two-sample test. Also benchmark on prompts resembling the real workload — the
per-workload spread here (tool calls 121 vs prose 100 on the same config) is larger than
most config deltas, so a single unrepresentative prompt can invert a conclusion.
