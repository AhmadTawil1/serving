"""Launch TensorRT-LLM's OpenAI-compatible server for a pre-built engine,
wait for it to report healthy, run a fixed warmup, and report the
concurrency-1 sanity number -- the TensorRT-LLM twin of `serve_vllm.py`,
sharing every piece of lifecycle logic via `serving/serve_common.py` so the
only framework-specific code here is the launch command itself.

GPU-only, Linux-only, and additionally requires a pre-built engine directory
(`scripts/build_trtllm_engine.py`) -- there is no from-HF-checkpoint
shortcut here, because the engine build itself, timed and logged, is the
Day 20 deliverable in its own right (§6: "engine build time is a reported
metric, not overhead").

Default port is 8001, not 8000, so a vLLM server (port 8000) and a
TensorRT-LLM server can run side by side on the same GPU host -- required
for the Day 20 parity check (`scripts/check_parity.py`), which needs both
endpoints live at once to compare their greedy output on the same prompt.

Usage, on the GPU host, inside TensorRT-LLM's own environment:

    python scripts/serve_trtllm.py --engine-dir /path/to/engine \\
        --card A100-SXM4-40GB --precision fp16 --port 8001
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from serving import config, serve_common  # noqa: E402

_LABEL = "serve_trtllm"


def launch(engine_dir: str, port: int) -> subprocess.Popen:
    """`trtllm-serve` exposes an OpenAI-compatible endpoint directly from a
    built engine directory. Exact flag names are provisional -- confirmed
    against whatever TensorRT-LLM version the Day 18 spike
    (`scripts/trtllm_spike.sh`) actually installs (PyPI listed up to 1.2.1
    as of 5 Aug 2026, per LOG.md Day 1). A wrong flag here fails loudly at
    launch, which is the "record every refusal with its exact error"
    behaviour §2.3 wants, not something to paper over with a try/except.
    """
    cmd = [
        "trtllm-serve",
        engine_dir,
        "--host",
        "0.0.0.0",
        "--port",
        str(port),
        "--max_batch_size",
        str(max(config.CONCURRENCY_LEVELS)),  # top of the concurrency axis (§3)
    ]
    print(f"[{_LABEL}] engine_dir={engine_dir}")
    print(f"[{_LABEL}] launching:", " ".join(cmd))
    return subprocess.Popen(cmd)


def main() -> None:
    serve_common.require_gpu_host(_LABEL)

    ap = argparse.ArgumentParser()
    ap.add_argument("--engine-dir", required=True)
    ap.add_argument("--card", required=True, choices=config.CARDS)
    ap.add_argument("--precision", default="fp16", choices=config.PRECISIONS)
    ap.add_argument("--port", type=int, default=8001)
    args = ap.parse_args()

    if not Path(args.engine_dir).exists():
        serve_common.fail(
            _LABEL,
            f"engine dir does not exist: {args.engine_dir} -- run "
            f"scripts/build_trtllm_engine.py first",
        )

    proc = launch(args.engine_dir, args.port)
    try:
        serve_common.wait_healthy(_LABEL, args.port)
        serve_common.warmup(_LABEL, args.port)
        serve_common.sanity_check(_LABEL, args.port)
        print(f"[{_LABEL}] server is up -- Ctrl+C to stop")
        proc.wait()
    except KeyboardInterrupt:
        pass
    finally:
        proc.terminate()


if __name__ == "__main__":
    main()
