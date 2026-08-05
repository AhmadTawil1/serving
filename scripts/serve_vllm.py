"""Launch vLLM's OpenAI-compatible server for the study model, wait for it to
report healthy, run a fixed warmup, and report the concurrency-1 sanity
number.

GPU-only, Linux-only -- vLLM has no Windows wheel and needs a CUDA device.
`serve_common.require_gpu_host` refuses to run anywhere else with a clear
message rather than a raw import/subprocess traceback, per the "check the
device" rule for this project (dev box is CPU-only; real runs are on the
rented A100/L4 in Colab, per `configs/serve-env.md`). This script is written
and structurally exercised on the dev box and *run* on the GPU host.

Usage, on the GPU host, inside vLLM's own environment:

    python scripts/serve_vllm.py --card A100-SXM4-40GB --precision fp16 --port 8000
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from serving import config, pins, serve_common  # noqa: E402

_LABEL = "serve_vllm"

# Flag names for vLLM's `--quantization` are provisional -- confirmed against
# the installed vLLM version on Day 22 when the precision ladder actually
# runs (§4.4). A wrong flag here should fail loudly at launch, which is
# exactly the "record every refusal with its exact error" behaviour §2.3
# wants, not something to paper over with a try/except.
_DTYPE_FLAGS = {
    "fp16": ["--dtype", "float16"],
    "fp8": ["--quantization", "fp8"],
    "int8": ["--quantization", "smoothquant"],
    "int4": ["--quantization", "awq"],
}


# Prefix caching is OFF by default for this study, decided 5 Aug 2026 after
# the first A100 curve (LOG.md Day 5). vLLM enables it by default; with only
# 8 frozen prompts and ~2,900 requests per window, the hit rate reached
# **96.9%** and GPU KV-cache usage sat at **5.2%** at concurrency 128. Nearly
# every request skipped prefill, so the reported TTFT was a cache lookup
# rather than a 512-token prefill, and the card was never loaded.
#
# Two reasons it is off rather than merely disclosed:
#   1. **Comparability.** TensorRT-LLM's prefix-reuse default is not vLLM's.
#      An unmatched framework default is a confound, not a finding, and §3
#      lists every such setting as held-fixed control.
#   2. **The axis has to bite.** With prefill cached the A100 showed zero
#      queue depth at concurrency 128, so no crossover can exist anywhere in
#      the swept range -- the study would measure nothing by construction.
#
# The caching-ON curve is kept as a real data point
# (`results/vllm_a100_fp16_prefixcache_on.jsonl`), not discarded.
_PREFIX_CACHE_DEFAULT = False


def launch(
    card: str, precision: str, port: int, prefix_caching: bool = _PREFIX_CACHE_DEFAULT
) -> subprocess.Popen:
    revision = pins.revision_for(config.STUDY_MODEL)
    cmd = [
        "vllm",
        "serve",
        config.STUDY_MODEL,
        "--revision",
        revision,
        "--max-model-len",
        str(config.MAX_MODEL_LEN),
        "--port",
        str(port),
        *_DTYPE_FLAGS[precision],
        # Stated explicitly in both directions rather than relying on the
        # framework default, so the launch command in the log always records
        # which regime the run was in.
        "--enable-prefix-caching" if prefix_caching else "--no-enable-prefix-caching",
    ]
    print(f"[{_LABEL}] card={card} precision={precision} revision={revision}")
    print(f"[{_LABEL}] prefix_caching={prefix_caching}")
    print(f"[{_LABEL}] launching:", " ".join(cmd))
    return subprocess.Popen(cmd)


def main() -> None:
    serve_common.require_gpu_host(_LABEL)

    ap = argparse.ArgumentParser()
    ap.add_argument("--card", required=True, choices=config.CARDS)
    ap.add_argument("--precision", default="fp16", choices=config.PRECISIONS)
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument(
        "--prefix-caching",
        action="store_true",
        default=_PREFIX_CACHE_DEFAULT,
        help=(
            "enable vLLM prefix caching. OFF by default for this study -- see "
            "the _PREFIX_CACHE_DEFAULT comment and LOG.md Day 5"
        ),
    )
    args = ap.parse_args()

    # Before launching, not after: a server already on this port would make
    # every later health check meaningless (LOG.md Day 8).
    serve_common.assert_port_free(_LABEL, args.port)

    proc = launch(args.card, args.precision, args.port, args.prefix_caching)
    try:
        serve_common.wait_healthy(_LABEL, args.port, proc=proc)
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
