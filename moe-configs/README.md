# moe-configs — retained for reference, NOT mounted

**This directory is deliberately not mounted into the vLLM container**, and
`VLLM_TUNED_CONFIG_FOLDER` has been removed from `docker-compose.yml`.

The single file here is a pre-tuned Triton config for vLLM's fused-MoE GEMM kernel:

```
E=256,N=512,device_name=NVIDIA_GB10,dtype=fp8_w8a8,block_shape=[128,128].json
```

## Why it was disabled

Read the filename: `dtype=fp8_w8a8`. The deployed model is **NVFP4**, not FP8. vLLM uses
the filename as a lookup key, so this config **never matched and was never loaded** — it had
been inert since the day it was added.

Renaming it to the key vLLM actually looks for made things worse. The config loaded, and the
engine crashed on startup:

```
triton.runtime.errors.OutOfResources: shared memory,
Required: 294912, Hardware limit: 101376
```

The tile parameters were tuned for a GPU with roughly 3x GB10's shared memory per SM. **The
filename mismatch had been silently protecting the stack from a config that would have
crashed it.**

Keeping it mounted is therefore a latent hazard rather than a no-op: if a future vLLM release
loosens its matching rules, a file that does nothing today becomes one that takes the server
down at startup.

The file is kept in the repository because the mechanism is worth documenting and because it
is a useful worked example of a config that appears to be helping and is not.

## If you want a tuned MoE config for this hardware

Generate one with vLLM's own `benchmark_moe.py` against **this** GPU and **this** dtype, then
mount it. Do not reuse a config from another machine or another quantization — verify the
`dtype=` and `device_name=` in the filename match your checkpoint before trusting it.

Note this only ever covered the Triton fused-MoE GEMM. vLLM runs other startup tuning passes
regardless, so it was never the whole story of the cold start.

See [`../IMPLEMENTATION_GUIDE.md`](../IMPLEMENTATION_GUIDE.md#moe-configs) for the longer
discussion.
