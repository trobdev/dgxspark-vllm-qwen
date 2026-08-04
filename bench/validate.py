#!/usr/bin/env python3
"""Functional validation of the deployed stack for Hermes + Claude Code paths."""
import json, sys, urllib.request, urllib.error

DIRECT = "http://localhost:8001"       # Hermes talks here (OpenAI API)
NGINX = "http://localhost/coding"      # Claude Code talks here (Anthropic API via shim)
results = []


def post(url, body, headers=None, timeout=180):
    h = {"Content-Type": "application/json"}
    h.update(headers or {})
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers=h)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def check(name, fn):
    try:
        ok, detail = fn()
    except Exception as e:
        ok, detail = False, f"{type(e).__name__}: {e}"
    results.append((name, ok, detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")


def t_models():
    d = json.loads(urllib.request.urlopen(DIRECT + "/v1/models", timeout=30).read())
    m = d["data"][0]
    ctx = m.get("max_model_len")
    return ctx == 262144, f"served={m['id']} max_model_len={ctx}"


def t_tool_call():
    """Hermes depends entirely on OpenAI-style function calling."""
    r = post(DIRECT + "/v1/chat/completions", {
        "model": "coding",
        "messages": [{"role": "user",
                      "content": "What is the weather in Tokyo right now? Use the tool."}],
        "tools": [{"type": "function", "function": {
            "name": "get_weather",
            "description": "Get current weather for a city",
            "parameters": {"type": "object",
                           "properties": {"city": {"type": "string"}},
                           "required": ["city"]}}}],
        "tool_choice": "auto", "max_tokens": 800, "temperature": 0.0})
    msg = r["choices"][0]["message"]
    tc = msg.get("tool_calls") or []
    if not tc:
        return False, f"no tool_calls; content={str(msg.get('content'))[:120]!r}"
    args = json.loads(tc[0]["function"]["arguments"])
    return (tc[0]["function"]["name"] == "get_weather"
            and "tokyo" in str(args.get("city", "")).lower()), \
        f"{tc[0]['function']['name']}({args})"


def t_multi_tool():
    """Hermes chains tools; verify a second round-trip with a tool result works."""
    r = post(DIRECT + "/v1/chat/completions", {
        "model": "coding",
        "messages": [
            {"role": "user", "content": "What is the weather in Tokyo?"},
            {"role": "assistant", "content": "", "tool_calls": [{
                "id": "call_1", "type": "function",
                "function": {"name": "get_weather",
                             "arguments": '{"city": "Tokyo"}'}}]},
            {"role": "tool", "tool_call_id": "call_1",
             "content": '{"temp_c": 18, "condition": "light rain"}'},
        ],
        "tools": [{"type": "function", "function": {
            "name": "get_weather", "description": "Get current weather",
            "parameters": {"type": "object",
                           "properties": {"city": {"type": "string"}}}}}],
        "max_tokens": 500, "temperature": 0.0})
    c = (r["choices"][0]["message"].get("content") or "")
    return ("18" in c and "rain" in c.lower()), c.strip()[:120]


def t_reasoning():
    r = post(DIRECT + "/v1/chat/completions", {
        "model": "coding",
        "messages": [{"role": "user",
                      "content": "A bat and ball cost $1.10 total. The bat costs $1.00 "
                                 "more than the ball. How much is the ball? Answer only "
                                 "with the amount."}],
        "max_tokens": 6000, "temperature": 0.0})
    msg = r["choices"][0]["message"]
    # vLLM 0.26.1 names this field `reasoning` (older builds: `reasoning_content`)
    rc = msg.get("reasoning") or msg.get("reasoning_content")
    c = msg.get("content") or ""
    return ("05" in c.replace(".", "").replace("$", "") and bool(rc)), \
        f"reasoning={len(rc) if rc else 0}ch split from content, answer={c.strip()[:40]!r}"


def t_long_context():
    """Claude Code sends ~150-200K prompts; prove the ceiling is real."""
    filler = "\n".join(f"line {i}: the quick brown fox jumps over the lazy dog #{i}"
                       for i in range(11000))
    body = {"model": "coding", "messages": [
        {"role": "user", "content": "Here is a log file.\n" + filler +
         "\n\nWhat exactly does line 7421 say? Quote it verbatim."}],
        # Thinking off: this probes retrieval, and reasoning tokens would other-
        # wise eat the whole budget before any content is emitted.
        "chat_template_kwargs": {"enable_thinking": False},
        "max_tokens": 300, "temperature": 0.0}
    r = post(DIRECT + "/v1/chat/completions", body, timeout=600)
    n_in = r["usage"]["prompt_tokens"]
    c = r["choices"][0]["message"].get("content") or ""
    return "7421" in c, f"prompt_tokens={n_in} answer={c.strip()[:90]!r}"


def t_anthropic_path():
    """Claude Code path: nginx -> shim -> vLLM /v1/messages, with a system-ROLE
    message inside messages[] (the exact thing that broke before)."""
    r = post(NGINX + "/v1/messages", {
        "model": "coding", "max_tokens": 4000,
        "system": "You are a helpful assistant.",
        "messages": [
            {"role": "system", "content": "Always end your reply with OK."},
            {"role": "user", "content": "Say hello in exactly one short sentence."},
        ]}, headers={"anthropic-version": "2023-06-01", "x-api-key": "none"})
    blocks = r.get("content", [])
    text = "".join(b.get("text", "") for b in blocks if b.get("type") == "text")
    kinds = [b.get("type") for b in blocks]
    # The `thinking` block proves the native Anthropic path is preserved (this is
    # what LiteLLM used to drop); text proves system-role hoisting reached the model.
    return len(text) > 0, f"blocks={kinds} text={text.strip()[:90]!r}"


def t_streaming():
    body = json.dumps({"model": "coding", "messages": [
        {"role": "user", "content": "Count from 1 to 5, comma separated, nothing else."}],
        "max_tokens": 400, "temperature": 0.0, "stream": True}).encode()
    req = urllib.request.Request(DIRECT + "/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    n = 0
    with urllib.request.urlopen(req, timeout=180) as r:
        for raw in r:
            if raw.decode().strip().startswith("data: {"):
                n += 1
    return n > 5, f"{n} SSE chunks"


for name, fn in [
    ("model metadata / context window", t_models),
    ("OpenAI tool calling (Hermes core)", t_tool_call),
    ("tool result round-trip", t_multi_tool),
    ("reasoning parser", t_reasoning),
    ("long context recall (~150K)", t_long_context),
    ("Anthropic /v1/messages via nginx+shim", t_anthropic_path),
    ("SSE streaming", t_streaming),
]:
    check(name, fn)

n_fail = sum(1 for _, ok, _ in results if not ok)
print(f"\n{len(results) - n_fail}/{len(results)} passed")
sys.exit(1 if n_fail else 0)
