#!/usr/bin/env python3
"""Sweep speculative-decode configs: rewrite docker-compose.yml, recreate the
vLLM container, wait for health, benchmark. Restores the original file at exit."""
import json, os, re, shutil, subprocess, sys, time, urllib.request, pathlib

HERE = pathlib.Path(__file__).resolve().parent
COMPOSE = pathlib.Path(os.environ.get("COMPOSE_FILE", HERE.parent / "docker-compose.yml"))
SCRATCH = pathlib.Path(os.environ.get("BENCH_OUTDIR", HERE))
ORIG = SCRATCH / "docker-compose.yml.orig"

DFLASH = "/models/Qwen3.6-35B-A3B-DFlash"
CONFIGS = [
    ("dflash8", {"method": "dflash", "model": DFLASH, "num_speculative_tokens": 8}),
    ("dflash4", {"method": "dflash", "model": DFLASH, "num_speculative_tokens": 4}),
    ("mtp3", {"method": "mtp", "num_speculative_tokens": 3, "moe_backend": "triton"}),
    ("nospec", None),
]

SPEC_BLOCK = re.compile(
    r'^[ \t]*-[ \t]*"--speculative-config"[ \t]*\n[ \t]*-[ \t]*\'[^\n]*\'[ \t]*\n',
    re.M)


def write_config(cfg):
    text = ORIG.read_text()
    if cfg is None:
        new = SPEC_BLOCK.sub("", text)
        assert new != text, "failed to strip speculative-config block"
    else:
        repl = ('      - "--speculative-config"\n'
                "      - '%s'\n" % json.dumps(cfg, separators=(",", ":")))
        new, n = SPEC_BLOCK.subn(lambda _: repl, text)
        assert n == 1, f"expected 1 spec block, replaced {n}"
    COMPOSE.write_text(new)


def recreate():
    subprocess.run(["docker", "compose", "up", "-d", "--force-recreate", "vllm-coding"],
                   cwd=COMPOSE.parent, check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)


def wait_healthy(timeout=900):
    t0 = time.time()
    while time.time() - t0 < timeout:
        state = subprocess.run(
            ["docker", "inspect", "vllm-coding", "--format",
             "{{.State.Health.Status}}|{{.State.Status}}"],
            capture_output=True, text=True).stdout.strip()
        health, status = (state.split("|") + [""])[:2]
        if health == "healthy":
            return True, time.time() - t0
        if status in ("exited", "dead"):
            return False, time.time() - t0
        time.sleep(10)
    return False, time.time() - t0


def crash_reason():
    log = subprocess.run(["docker", "logs", "--tail", "40", "vllm-coding"],
                         capture_output=True, text=True)
    return (log.stdout + log.stderr)[-1500:]


def main():
    if not ORIG.exists():
        shutil.copy(COMPOSE, ORIG)
        print(f"saved original -> {ORIG}")

    try:
        for label, cfg in CONFIGS:
            print(f"\n{'='*60}\n=== {label}: {json.dumps(cfg) if cfg else 'no spec decode'}",
                  flush=True)
            write_config(cfg)
            recreate()
            ok, secs = wait_healthy()
            if not ok:
                print(f"!!! {label} FAILED to become healthy after {secs:.0f}s")
                print(crash_reason(), flush=True)
                continue
            print(f"    healthy in {secs:.0f}s", flush=True)
            subprocess.run([sys.executable, str(SCRATCH / "bench_hermes.py"), label, "5"],
                           check=False)
    finally:
        shutil.copy(ORIG, COMPOSE)
        print("\ndocker-compose.yml restored to pre-sweep state "
              "(container still running the LAST swept config).")


if __name__ == "__main__":
    main()
