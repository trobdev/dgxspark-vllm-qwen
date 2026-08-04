#!/usr/bin/env python3
"""Benchmark vLLM against Hermes-Agent-representative workloads.

Measures pure decode rate (excludes prefill/TTFT) via streaming, plus
speculative-decode acceptance from vLLM's Prometheus metrics.

Usage: bench_hermes.py <label> [trials]
"""
import json, os, sys, time, statistics, urllib.request, re
from concurrent.futures import ThreadPoolExecutor

BASE = os.environ.get("VLLM_BASE", "http://localhost:8001")
MAXTOK = 512

# Four workload archetypes matching how Hermes actually drives this model.
PROMPTS = {
    # 1. Agentic tool call: highly structured JSON — best case for spec decode
    "tool_call": [
        {"role": "system", "content": "You are Hermes, a personal assistant with tools. "
         "Available tools: file(read/write/list), terminal(run), web(search/fetch), "
         "todo(add/list/complete), cronjob(create/list). Respond ONLY with a JSON array "
         "of tool invocations, each {\"tool\":..., \"action\":..., \"args\":{...}}."},
        {"role": "user", "content": "Check my project directory ~/proj for a README, "
         "read it, search the web for the latest version of its main dependency, then "
         "add a todo to upgrade it and schedule a weekly check."},
    ],
    # 2. Code edit: reproduces surrounding context — predictable
    "code_edit": [
        {"role": "system", "content": "You are a careful software engineer."},
        {"role": "user", "content": """Add input validation and a docstring to this function. Return the complete updated file.

```python
import json
from pathlib import Path

def load_config(path, defaults=None):
    data = json.loads(Path(path).read_text())
    if defaults:
        for k, v in defaults.items():
            data.setdefault(k, v)
    return data

def save_config(path, data):
    Path(path).write_text(json.dumps(data, indent=2))
```"""},
    ],
    # 3. Content development: novel prose — worst case for spec decode
    "prose": [
        {"role": "system", "content": "You are a skilled technical writer."},
        {"role": "user", "content": "Write an original 400-word blog introduction about "
         "why local inference is becoming practical for small teams. Avoid cliches and "
         "do not reuse standard marketing phrasing."},
    ],
    # 4. General PA reasoning: mixed
    "reasoning": [
        {"role": "system", "content": "You are a helpful, concise personal assistant."},
        {"role": "user", "content": "I have three deadlines: a client report Friday "
         "(8h of work), a tax filing next Wednesday (2h), and a conference talk in three "
         "weeks (20h). I have about 4 focused hours a day. Lay out a concrete schedule "
         "and flag any risk of slipping."},
    ],
}


def one(name):
    """Run one streaming request; return (decode_tok_s, ttft, n_out)."""
    body = json.dumps({
        "model": "coding", "messages": PROMPTS[name], "max_tokens": MAXTOK,
        "temperature": 0.0, "stream": True,
        "stream_options": {"include_usage": True},
    }).encode()
    req = urllib.request.Request(BASE + "/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    first = None
    n_out = 0
    with urllib.request.urlopen(req, timeout=600) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data: "):
                continue
            payload = line[6:]
            if payload == "[DONE]":
                break
            chunk = json.loads(payload)
            if chunk.get("usage"):
                n_out = chunk["usage"]["completion_tokens"]
            ch = chunk.get("choices") or []
            if ch and (ch[0].get("delta") or {}).get("content") is not None:
                if first is None:
                    first = time.perf_counter()
    t1 = time.perf_counter()
    if first is None or n_out < 2:
        return None
    return ((n_out - 1) / (t1 - first), first - t0, n_out)


def metrics():
    """Pull spec-decode acceptance counters."""
    try:
        with urllib.request.urlopen(BASE + "/metrics", timeout=10) as r:
            txt = r.read().decode()
    except Exception:
        return {}
    out = {}
    for key in ("spec_decode_num_accepted_tokens_total",
                "spec_decode_num_draft_tokens_total"):
        m = re.findall(r"^vllm:%s\{[^}]*\}\s+([0-9.e+]+)$" % key, txt, re.M)
        if m:
            out[key] = sum(float(x) for x in m)
    return out


def main():
    label = sys.argv[1] if len(sys.argv) > 1 else "run"
    trials = int(sys.argv[2]) if len(sys.argv) > 2 else 5

    print(f"### {label}")
    # Warm up (also populates prefix cache so prefill is out of the picture)
    for n in PROMPTS:
        one(n)

    m0 = metrics()
    result = {"label": label, "c1": {}, "c4": {}}

    # --- concurrency 1: per-workload decode rate ---
    for name in PROMPTS:
        rates = []
        for _ in range(trials):
            r = one(name)
            if r:
                rates.append(r[0])
        med = statistics.median(rates)
        spread = (max(rates) - min(rates)) / med * 100
        result["c1"][name] = {"median": med, "n": len(rates),
                              "spread_pct": spread, "all": rates}
        print(f"  c=1 {name:10s} {med:7.1f} tok/s   (spread {spread:4.1f}%)")

    c1_all = [v["median"] for v in result["c1"].values()]
    print(f"  c=1 OVERALL median-of-medians: {statistics.median(c1_all):.1f} tok/s")

    # --- concurrency 4: aggregate throughput (delegation / sub-agents) ---
    names = list(PROMPTS)
    aggs = []
    import os
    for _ in range(int(os.environ.get("BENCH_C4_ROUNDS", "3"))):
        t0 = time.perf_counter()
        with ThreadPoolExecutor(max_workers=4) as ex:
            rs = list(ex.map(one, names))
        el = time.perf_counter() - t0
        tot = sum(r[2] for r in rs if r)
        aggs.append(tot / el)
    result["c4"]["agg"] = statistics.median(aggs)
    print(f"  c=4 aggregate throughput: {statistics.median(aggs):.1f} tok/s "
          f"(runs: {', '.join(f'{a:.0f}' for a in aggs)})")

    # --- spec decode acceptance over this run ---
    m1 = metrics()
    if m0 and m1:
        acc = m1.get("spec_decode_num_accepted_tokens_total", 0) - \
              m0.get("spec_decode_num_accepted_tokens_total", 0)
        drf = m1.get("spec_decode_num_draft_tokens_total", 0) - \
              m0.get("spec_decode_num_draft_tokens_total", 0)
        if drf > 0:
            result["acceptance"] = acc / drf
            print(f"  spec-decode acceptance: {acc/drf*100:.1f}% "
                  f"({acc:.0f}/{drf:.0f} draft tokens)")
    print()
    outdir = os.environ.get("BENCH_OUTDIR", os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(outdir, f"res_{label}.json"), "w") as f:
        json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
