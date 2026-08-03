# v2 Rebuild Plan — vLLM 0.26.x + unsloth NVFP4

**Status:** scoped, not started
**Scoped:** 2026-08-03
**Goal:** move the stack from vLLM `0.22.1rc1.dev` to `0.26.x`, and (separately) from the
NVIDIA NVFP4 checkpoint to the unsloth NVFP4 checkpoint with calibrated KV-cache scales.

---

---

## ⛔ Hard constraint — no driver updates

**Never perform a GPU driver update as part of this work.** If any step turns out to
require a driver newer than the currently installed **`580.173.02`**, **stop and alert the
operator.** Do not attempt a workaround, a partial upgrade, or a container-side shim to
dodge it. Driver updates are performed by the machine's owner only, and an in-progress
migration is not a reason to proceed.

### Driver compatibility — ✅ VERIFIED GO (2026-08-03)

This was the go/no-go gate. It clears, and by a comfortable margin — **v2 moves toward the
installed driver, not away from it.**

| Evidence | Finding |
|---|---|
| Upstream base image (`ARG CUDA_IMAGE`, Dockerfile line 5) | **`nvidia/cuda:13.0.2-devel-ubuntu24.04`** — *lower* than the `13.2.0` currently running |
| Driver's natively supported CUDA (`nvidia-smi`) | **13.0** — exactly matches the v2 base image |
| Upstream `torch==2.11.0` install index | **`cu130`** — same CUDA 13.0 target as the current `torch 2.10.0+cu130`. A PyTorch bump, not a CUDA bump. |
| Current runtime proof | The running container already executes **CUDA 13.2** on this 580.x driver via forward compatibility (`cuda-compat-13-2`, `libcuda.so.595.45.04`, confirmed loaded by the live vLLM process). v2's 13.0.2 is strictly less demanding. |

Installed driver: `580.173.02` (NVRM build 2026-06-23). Host userspace lib:
`libcuda.so.580.173.02`.

> Earlier revisions of this document cited `580.159.03`. That reading predated the
> mid-July host reboots (kernel `6.17.0-1021-nvid` → `6.17.0-1026-nvid`), during which the
> host updated itself. Both are 580.x and both natively support CUDA 13.0, so the
> conclusion is unchanged.

**Residual risk (cannot be pre-verified):** this establishes that the *declared* CUDA and
torch targets need no driver update. It does not prove that some transitive dependency (a
FlashInfer cubin, a specific CUTLASS DSL build) won't demand newer CUDA userspace at
runtime. Such a failure surfaces **inside the container** as a CUDA-init or PTX error, not
as a host problem. Response: stop and report — the forward-compat layer is the only
sanctioned mitigation, never a host driver change.

---

## 0. Why

Three independent wins motivate this, in rough order of value:

1. **Calibrated FP8 KV cache.** The unsloth checkpoint ships real `k_scale`/`v_scale`
   tensors (verified: 10 + 10, matching the 10 full-attention layers). The NVIDIA
   checkpoint ships none — which is exactly why `--kv-cache-dtype fp8` was rejected
   previously (vLLM falls back to `scale=1.0` and silently clips values above FP8
   E4M3's max of 448). With real scales that objection is gone, and FP8 KV cache
   roughly **halves** per-token KV cost (~20 KB → ~10 KB/token).
2. **Four minor versions of upstream fixes**, including several that target this exact
   configuration (mamba + MTP prefix caching, faster model loading, memory-pressure
   handling).
3. **Corrected tool-call parser.** Upstream's maintained recipe for this model uses
   `qwen3_xml`, not the `qwen3_coder` currently in `docker-compose.yml`. This may be a
   live correctness bug today, independent of v2.

Native NVFP4 compute (FlashInfer b12x / vLLM PR #40082) is **explicitly out of scope** —
see [Deferred](#7-deferred-explicitly-not-in-v2).

---

## 1. Rollback plan — ✅ COMPLETED 2026-08-03

> Artifacts live in `~/llm-stack-backups/` (**not** `/data`, which is root-owned).
> See `~/llm-stack-backups/RESTORE.md` for the restore runbook.
> The image export is uncompressed (19 GB) rather than gzipped — disk is abundant
> (2.8 TB free) and it avoids a slow ARM compression pass.

> **The current image exists nowhere but this machine.** `vllm-node:latest` was built
> locally and never pushed to a registry (`RepoDigests` is empty). A rebuild reuses the
> `:latest` tag, so the moment a new build succeeds the known-good image is untagged and
> eligible for pruning. There is no re-download path.

### Current known-good state (restore target)

| Field | Value |
|---|---|
| Image | `vllm-node:latest` |
| Image ID | `sha256:daecb67b10c8b428e1c307a8beb2cce4bc12bd1e36215cb3b7c7f93d70b9e8c6` |
| Size | 19.3 GB |
| Built | 2026-05-30T20:48:52Z |
| vLLM | `0.22.1rc1.dev22+g3fd9d2d35.d20260530.cu132` |
| vLLM commit | `3fd9d2d35714e80b4cb3fcd3c408a0398fa2525f` |
| FlashInfer commit | `fc12ef21` (wheels: 0.6.12) |
| Build script commit | `c187912e23442f1868a58c86f28a644128ce01b2` |
| GPU arch | `12.1a` |
| Base image | `nvidia/cuda:13.2.0-devel-ubuntu24.04` |

### Preservation steps

```bash
# 1. Stop the tag from being overwritten (instant, free — do this before anything else)
docker tag vllm-node:latest vllm-node:v1-0.22.1rc1

# 2. Offline copy that survives `docker system prune`
#    (there is currently 35 GB reclaimable + 49 GB build cache, so a prune is plausible
#     during this work — an untagged image would not survive it)
mkdir -p ~/llm-stack-backups
docker save vllm-node:v1-0.22.1rc1 | gzip > ~/llm-stack-backups/vllm-node-v1-0.22.1rc1.tar.gz

# 3. The exact wheels that built the current image (~1.16 GB).
#    build-and-copy.sh's try_download_wheels() writes into ./wheels/ and can clobber these.
cp -a ~/spark-vllm-docker/wheels ~/llm-stack-backups/wheels-v1-0.22.1rc1

# 4. Local Dockerfile modifications — UNCOMMITTED, would be lost/conflicted by `git pull`
cd ~/spark-vllm-docker && git diff > ~/llm-stack-backups/spark-vllm-docker-local-mods.patch

# 5. Commit the pending compose changes on the stack repo
cd ~/llm-stack && git add docker-compose.yml && git commit -m "..."

# 6. Work on a branch, not main
git checkout -b v2-vllm-026
```

**Do not delete** `/data/models/Qwen3.6-35B-A3B-NVFP4` (22 GB). Disk is 2.9 TB free; the
unsloth checkpoint (26.5 GB) sits alongside it.

### Restore procedure

```bash
docker tag vllm-node:v1-0.22.1rc1 vllm-node:latest
cd ~/llm-stack && git checkout main -- docker-compose.yml
docker compose up -d
```

If the local image was pruned:
`gunzip -c ~/llm-stack-backups/vllm-node-v1-0.22.1rc1.tar.gz | docker load` first.

**Time to roll back: ~6 minutes** (dominated by model reload).

---

## 2. Sequencing — two deployments, not one

The image upgrade and the model swap are independent changes. Doing both at once means a
throughput regression or a broken tool call can't be attributed to either.

- **Phase A — image only.** Upgrade to vLLM 0.26.x while keeping the **NVIDIA**
  checkpoint (already on disk; upstream's recipe targets it directly). Validate. This is
  its own rollback point.
- **Phase B — model only.** Swap to the unsloth checkpoint and enable
  `--kv-cache-dtype fp8`. Validate again.

Costs roughly an extra hour; converts one ambiguous failure into two clean ones.

---

## 3. What changes

### 3a. Build repo (`~/spark-vllm-docker`)

Upstream is **~30 commits ahead** (`c187912..f7d6e3b`). Directly relevant:

| Commit | Why it matters here |
|---|---|
| `d704b84` | Improves prefix caching with **mamba models + MTP** — this exact config |
| `173b5f3`, `552b618` | Switched to `instanttensor` loading — targets the 5-6 min startup |
| `8a6a007` | `earlyoom` support — relevant to host memory-pressure concerns |
| `20fd867` | PyTorch expandable segments by default — memory fragmentation |
| `81a33f3`, `562ed29` | FlashInfer regression fixes |

**Conflict to resolve:** local `Dockerfile` pins `torch==2.10.0+cu130`; upstream now uses
`2.11.0`. Local also adds `--index-strategy unsafe-best-match` / `--extra-index-url` to
the wheel install steps. Decide whether upstream's 2.11.0 works on GB10 before discarding
the local pin — it was presumably pinned for a reason. (Both target `cu130`, so this is a
functional question, not a driver-compatibility one.)

**Structural change:** upstream's Dockerfile grew from 341 → 994 lines and now
parameterizes the base image as `ARG CUDA_IMAGE`, defaulting to
`nvidia/cuda:13.0.2-devel-ubuntu24.04` (down from the hardcoded `13.2.0` in the current
build). Expect the local-mods patch to need manual reapplication rather than a clean
`git apply` — the surrounding context has moved substantially.

### 3b. vLLM version

`0.22.1rc1.dev` → **`v0.26.0`** (stable, 2026-07-27). Minimum for the unsloth checkpoint
is `0.25.0` (2026-07-11).

### 3c. Serve flags

Deltas implied by upstream's `recipes/qwen3.6-35b-a3b-nvfp4.yaml`. Note that recipe is
written for a 2-node cluster (`tensor_parallel: 2`, `gpu_memory_utilization: 0.4`) — the
values below must **not** be copied verbatim for this single-node TP=1 setup.

| Change | Rationale |
|---|---|
| `--tool-call-parser qwen3_coder` → **`qwen3_xml`** | Upstream uses `qwen3_xml` for Qwen3.6. See also `mods/fix-qwen3.6-chat-template/`. Possible live bug today. |
| **add** `--kv-cache-dtype fp8` | Phase B only — requires unsloth's calibrated scales |
| **add** `--moe-backend marlin` | Explicit backend selection; flag does not exist in 0.22.1rc1 |
| **add** `--attention-backend flashinfer` | |
| **add** `--load-format fastsafetensors` | Attacks the 5-6 min startup |
| **add** `--async-scheduling`, `--enable-chunked-prefill` | |
| **add** env `VLLM_MARLIN_USE_ATOMIC_ADD=1` | |
| **verify** `--language-model-only` | **Unconfirmed** to survive to 0.26.0 — check `--help` in the new image before first boot |
| **verify** `--speculative-config` MTP JSON shape | Format may have shifted across four minor versions |

Keep as-is: `--gpu-memory-utilization 0.75`, `--max-model-len 262144`, `--max-num-seqs 32`,
`--reasoning-parser qwen3`, `--enable-prefix-caching`, `--enable-auto-tool-choice`,
`--tensor-parallel-size 1`.

### 3d. Model (Phase B)

| | current (NVIDIA) | v2 (unsloth) |
|---|---|---|
| Repo | `nvidia/Qwen3.6-35B-A3B-NVFP4` | `unsloth/Qwen3.6-35B-A3B-NVFP4` |
| On-disk | 22 GB | 26.5 GB (6 shards) |
| Tensors | 124,468 | 112,020 |
| `k_scale`/`v_scale` | **none** | **10 + 10** |
| `kv_cache_scheme` | absent | FP8, `static_minmax`, symmetric, per-tensor |
| MTP module | yes (19 tensors) | yes (19 tensors) |
| License | NVIDIA Open Model | Apache 2.0 |

Architecture is otherwise **identical** — same `Qwen3_5MoeForConditionalGeneration`, 40
layers, 256 experts / 8 active, `head_dim` 256, `full_attention_interval` 4, 262144 native
context. Fewer quantized tensors at larger size implies unsloth left more layers in higher
precision, consistent with their claim of comparable accuracy.

Add to `download-models.sh` alongside the existing entry (do not replace it).

### 3e. Side fix — `moe-configs` is currently inert

vLLM logs:

```
Using default MoE config. Performance might be sub-optimal!
Config file not found at /moe-configs/E=256,N=512,device_name=NVIDIA_GB10.json
```

The mounted file is named
`E=256,N=512,device_name=NVIDIA_GB10,dtype=fp8_w8a8,block_shape=[128,128].json`, so it has
**never loaded**. Either rename to the expected form or regenerate for the new build.
Free performance available here.

### 3f. Docs

`README.md`, `SETUP_GUIDE.md`, `IMPLEMENTATION_GUIDE.md` and
`dgx_spark_vllm_stack_v2.svg` already carry drift from the 2026-07-03 config changes
(still document `131072` context, `0.85` utilization, "no speculative decoding", 60-80
tok/s). v2 compounds this. Reconcile in one pass at the end rather than incrementally.

---

## 4. Effort & time

| Phase | Est. | Risk |
|---|---|---|
| §1 backups | 30-45 min | none |
| Model download (26.5 GB, backgroundable) | 15-40 min | low |
| Image build — prebuilt wheels | 15-20 min | **high** |
| Image build — from source (fallback) | 30-60+ min | **high** |
| Config migration + first successful boot | 1-2 hr | **high** |
| Validation (§5) | ~1 hr | med |
| Docs (§3f) | 1-2 hr | low |

**Realistic total: half a day if smooth, a full day with debugging.** Not a quick swap.

---

## 5. Validation checklist

Run after **each** phase (A and B), against the previous phase's numbers:

- [ ] Container reaches `healthy`; no `ERROR`/`Traceback` in startup logs
- [ ] Startup wall-clock recorded (baseline: ~5-6 min; `fastsafetensors` should improve it)
- [ ] `nvidia-smi` process memory recorded (baseline: ~88.5-90.8 GB at util 0.75)
- [ ] Throughput, warm + uncontended, ≥3 trials (baseline: ~123-124 tok/s, 500-token gen)
- [ ] Marlin vs FlashInfer path confirmed from startup log warnings
- [ ] Tool calling works end-to-end (**specifically** re-test after the `qwen3_xml` change)
- [ ] `thinking` blocks survive the anthropic-shim → `/v1/messages` path
- [ ] Real Claude Code session via `claude-local` completes a multi-turn task
- [ ] Long-context request (>131072 tokens) succeeds — confirms the 262144 ceiling
- [ ] Phase B only: FP8 KV cache produces coherent output at long context (guard against
      the silent-clipping failure mode that motivated the original rejection)

---

## 6. Risk register

| Risk | Severity | Mitigation |
|---|---|---|
| Dockerfile hardcodes a patch for vLLM PR #35568 ("broken FP8 kernels"). Likely merged by 0.26.0; if the diff applies cleanly in neither direction, `git apply -v` fails and the build dies. | **High** | Check whether #35568 is in 0.26.0 and remove the patch block before building |
| Four minor versions of flag/API drift; `--language-model-only` unverified | **High** | Run `--help` in the new image *before* wiring it into compose |
| torch 2.10.0 (local pin) vs 2.11.0 (upstream) | Med | Test upstream 2.11.0 first; local pin is the fallback (`spark-vllm-docker-local-mods.patch`). **No driver implication** — both install from the `cu130` index. |
| A transitive dep (FlashInfer cubin, CUTLASS DSL) demands newer CUDA userspace than the driver provides | Med | Surfaces inside the container, not on the host. **Stop and report** — never a host driver change. Forward-compat (`cuda-compat`) is the only sanctioned mitigation. |
| MTP `--speculative-config` JSON shape may have changed | Med | Validate against 0.26.0 docs; `recipes/qwen3.6-35b-a3b-nvfp4-no-mtp.yaml` exists as a fallback |
| anthropic-shim untested against 0.26.0's `/v1/messages` | Med | Covered by the validation checklist; shim is stdlib-only and easy to patch |
| FP8 KV cache degrades quality subtly rather than loudly | Med | Phase B is separately reversible; long-context coherence check in §5 |
| Disk | Low | 2.9 TB free |

---

## 7. Deferred — explicitly not in v2

- **Native NVFP4 compute** (FlashInfer b12x, vLLM PR #40082, merged 2026-05-20). Upstream
  has an experimental `b12x` branch with an `--exp-b12x` flag. Deferred because: (a) the
  NVIDIA developer forum reports Marlin still *outperforms* the native path on GB10 today;
  (b) upstream's own maintained recipe for this model still specifies
  `--moe-backend marlin`; (c) a maintainer flagged an unresolved correctness concern
  (in-place mutation of a quantization scale tensor) during PR review. Revisit once the
  recipe itself switches away from Marlin.
- **Idle scale-down** via vLLM's `/sleep` + `/wake_up` (`--enable-sleep-mode`). Real and
  present in the current image, but gated behind `VLLM_SERVER_DEV_MODE=1`, needs a custom
  idle-watcher, and is untested against this Marlin + MTP + mamba-hybrid combination.
- **Migration to a Nemotron checkpoint.** Researched 2026-08-03; similar tier, no
  compelling reason to move.
