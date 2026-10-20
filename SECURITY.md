# Security

## Reporting a vulnerability

Please report security problems privately through GitHub's
[private vulnerability reporting](https://github.com/trobdev/dgxspark-vllm-qwen/security/advisories/new)
rather than in a public issue.

## Scope and threat model

This stack is built for a single developer or a trusted LAN. It is not designed to be
exposed to the internet.

- **What is protected:** nginx on port 80 requires `LLM_API_KEY` on every `/coding/` request.
- **What is not protected:** vLLM (`127.0.0.1:8001`) and the shim (`127.0.0.1:8003`) have
  no authentication. They are bound to loopback, so only processes on the host can reach
  them. Keep them that way.
- **TLS is off by default.** On an untrusted network, the key and your prompts cross the
  wire in plain text. Enable the HTTPS block in `nginx/default.conf.template`, or use a VPN.
- **Docker-published ports bypass ufw.** Don't rely on host firewall rules to restrict port 80.

Issues in vLLM, FlashInfer, or `eugr/spark-vllm-docker` should go to those projects.
