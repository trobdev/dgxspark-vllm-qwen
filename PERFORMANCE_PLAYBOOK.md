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
memory-limited box. Measured sweep (`num_speculative_tokens`, aggregate tok/s @ c=1):

```
2 -> 90.2      3 -> 108.1      5 -> 95.2
```

**3 is optimal here.** Acceptance falls 78.3% -> 64.7% going 3 -> 5, and the extra draft
passes stop paying for themselves. Do not copy a vendor's default — sweep it.

Important structural note: with `method: mtp` the **draft layer is unquantized BF16**, so
each draft pass costs real bandwidth. Draft-free methods (see §4) may beat it on this
platform precisely because they read *zero* extra weights.

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

- `--max-num-batched-tokens 8192`: **+7%** (101.3 -> 108.1 @ c=1). vLLM warns the implicit
  2048 starves spec-decode draft slots. Always set this when speculation is on.
- MoE backend choice (Marlin vs `flashinfer_b12x`): **measured a wash** at c=1-8. Do not
  spend time here unless operating at high concurrency.

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
