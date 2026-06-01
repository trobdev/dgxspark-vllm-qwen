#!/usr/bin/env python3
"""
anthropic-shim — normalize Claude Code's Anthropic requests for vLLM.

Claude Code injects a `system`-role message *inside* the `messages` array
(MCP instructions, skills list, IDE context). vLLM's native /v1/messages
endpoint strictly allows only user/assistant roles and 400s on it. This shim
hoists any system-role messages into the top-level `system` field and forwards
to vLLM's native /v1/messages — preserving the Anthropic format (and thus the
model's `thinking` blocks). All other traffic is proxied through untouched,
including streamed (SSE) responses, which are relayed chunk-by-chunk.

Dependency-free (stdlib only). Listens on :8080. Upstream from $VLLM_UPSTREAM.
"""
import json
import os
import sys
import urllib.request
import urllib.error
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

UPSTREAM = os.environ.get("VLLM_UPSTREAM", "http://vllm-coding:8000").rstrip("/")
LISTEN_PORT = int(os.environ.get("SHIM_PORT", "8080"))

# Headers we must not copy verbatim between hops.
HOP_BY_HOP = {
    "connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
    "te", "trailers", "transfer-encoding", "upgrade", "content-length",
}


def hoist_system(payload):
    """Move any role=='system' messages out of messages[] and append their
    content onto the top-level `system` field. Returns the (mutated) payload."""
    msgs = payload.get("messages")
    if not isinstance(msgs, list):
        return payload

    hoisted, kept = [], []
    for m in msgs:
        if isinstance(m, dict) and m.get("role") == "system":
            c = m.get("content")
            if isinstance(c, list):
                hoisted.extend(c)                       # already content blocks
            elif isinstance(c, str):
                hoisted.append({"type": "text", "text": c})
            elif c is not None:
                hoisted.append({"type": "text", "text": str(c)})
        else:
            kept.append(m)

    if not hoisted:
        return payload

    existing = payload.get("system")
    if existing is None:
        new_system = hoisted
    elif isinstance(existing, str):
        new_system = [{"type": "text", "text": existing}] + hoisted
    elif isinstance(existing, list):
        new_system = existing + hoisted
    else:
        new_system = hoisted

    payload["system"] = new_system
    payload["messages"] = kept
    return payload


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def _healthz(self):
        body = b"ok\n"
        self.send_response(200)
        self.send_header("content-type", "text/plain")
        self.send_header("content-length", str(len(body)))
        self.send_header("connection", "close")
        self.end_headers()
        self.wfile.write(body)

    def _proxy(self, method):
        if self.path == "/healthz":
            return self._healthz()

        # Read request body (if any) and normalize JSON Anthropic requests.
        length = int(self.headers.get("content-length", 0) or 0)
        body = self.rfile.read(length) if length else b""
        if body:
            try:
                payload = json.loads(body)
                body = json.dumps(hoist_system(payload)).encode("utf-8")
            except (ValueError, TypeError):
                pass  # not JSON — forward unchanged

        # Build the upstream request, copying through meaningful headers.
        req = urllib.request.Request(UPSTREAM + self.path, data=body if body else None, method=method)
        for k, v in self.headers.items():
            if k.lower() in HOP_BY_HOP or k.lower() == "host":
                continue
            req.add_header(k, v)
        if body:
            req.add_header("Content-Length", str(len(body)))

        # Forward and relay the response, streaming the body chunk-by-chunk.
        try:
            resp = urllib.request.urlopen(req)
        except urllib.error.HTTPError as e:
            resp = e  # relay upstream error responses (e.g. 400) verbatim
        except urllib.error.URLError as e:
            msg = json.dumps({"type": "error", "error": {"type": "upstream_error",
                              "message": f"shim could not reach vLLM: {e}"}}).encode()
            self.send_response(502)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(msg)))
            self.send_header("connection", "close")
            self.end_headers()
            self.wfile.write(msg)
            return

        self.send_response(resp.status)
        for k, v in resp.headers.items():
            if k.lower() in HOP_BY_HOP:
                continue
            self.send_header(k, v)
        # We stream with unknown length, so close the hop when done.
        self.send_header("connection", "close")
        self.end_headers()
        try:
            while True:
                chunk = resp.read(8192)
                if not chunk:
                    break
                self.wfile.write(chunk)
                self.wfile.flush()
        except BrokenPipeError:
            pass
        finally:
            resp.close()

    def do_POST(self):
        self._proxy("POST")

    def do_GET(self):
        self._proxy("GET")

    def log_message(self, *args):
        pass  # quiet; rely on nginx/vLLM logs


if __name__ == "__main__":
    print(f"anthropic-shim listening on :{LISTEN_PORT}, upstream {UPSTREAM}", flush=True)
    try:
        ThreadingHTTPServer(("0.0.0.0", LISTEN_PORT), Handler).serve_forever()
    except KeyboardInterrupt:
        sys.exit(0)
