# Benchmark & validation harness

Standard-library Python only. Used to produce the results in
[`../OPTIMIZATION_REPORT.md`](../OPTIMIZATION_REPORT.md).

## `bench_hermes.py <label> [trials]`

Measures **decode rate only** — streams via SSE, records time-to-first-token separately and
excludes it, so prefill and prefix-cache effects don't contaminate the number:

```
decode_tok_s = (completion_tokens - 1) / (t_last_token - t_first_token)
```

Runs four workload archetypes matching real agent usage (`tool_call`, `code_edit`, `prose`,
`reasoning`) — the spread between these on a *fixed* config is larger than most config
deltas, so benchmarking on one unrepresentative prompt can invert a conclusion. Reports
per-workload medians, a median-of-medians headline, concurrency-4 aggregate throughput, and
speculative acceptance differenced from vLLM's Prometheus counters. Warms up first (the
first run after a restart is 3-10x slower from JIT).

Env: `VLLM_BASE` (default `http://localhost:8001`), `BENCH_C4_ROUNDS` (default 3),
`BENCH_OUTDIR` (default: this directory). Writes `res_<label>.json` with raw per-trial samples.

```bash
python3 bench/bench_hermes.py baseline 14      # n=14, decision-grade
```

## `sweep.py`

Rewrites the `--speculative-config` block in `docker-compose.yml`, force-recreates the
container, polls until healthy, benchmarks, then moves to the next config. **Restores the
original compose file on exit** (including on crash). Edit `CONFIGS`, or import and override:

```python
import sweep
sweep.CONFIGS = [("dflash4", {"method": "dflash", "model": "...", "num_speculative_tokens": 4})]
sweep.main()
```

Note cold start is 341-461s per config, so a 4-config sweep takes ~40 minutes.

## `validate.py`

Seven functional checks across both API surfaces — run before committing any config change.
Exits non-zero on failure.

Covers: model metadata / context window, OpenAI tool calling, tool-result round-trip,
reasoning/content separation, ~240K-token context recall, the Anthropic `/v1/messages` path
through nginx + the normalizing shim (confirms `thinking` blocks survive), and SSE streaming.

Two gotchas it encodes, both of which cost debugging time:

- vLLM 0.26.1 names the reasoning field **`reasoning`**, not `reasoning_content`. Checking
  the old name makes a working parser look broken.
- This is a reasoning model: if `max_tokens` is smaller than the thinking block, `content`
  comes back **empty** — the budget is consumed before `</think>`. Use
  `chat_template_kwargs: {"enable_thinking": false}` for pure-retrieval calls.

## Statistical practice

The no-speculation control measures at **0.2-1.3% spread**, so the rig is precise to ~1%.
The larger variance seen with speculation enabled is acceptance-rate jitter, *not* instrument
error — meaning sub-10% differences are unresolved rather than absent, and are recoverable by
raising n. Use n>=14 and a two-sample test for anything that decides a deployment.
