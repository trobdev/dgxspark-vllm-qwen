---
name: Bug report
about: Something in the setup or the running stack doesn't work as documented
labels: bug
---

**What happened**

**What you expected**

**Which step** (e.g. README quick start step 4, SETUP_GUIDE Phase 2)

**Environment**
- Hardware (DGX Spark / OEM GB10 model):
- `nvidia-smi` driver version:
- vLLM / FlashInfer version — output of:
  `docker run --rm --runtime=nvidia vllm-node-v2:latest python3 -c "import vllm, flashinfer; print(vllm.__version__, flashinfer.__version__)"`
- Changes to `docker-compose.yml`, if any:

**Output of `python3 bench/validate.py`** (if the stack starts)

```
```

**Relevant logs** (`docker compose logs --tail 100 vllm-coding`)

```
```
