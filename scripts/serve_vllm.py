"""Launch vLLM's OpenAI-compatible server for the study model, wait for it to
report healthy, run a fixed warmup, and report Day 18's number: tokens/sec at
concurrency 1.

GPU-only, Linux-only -- vLLM has no Windows wheel and needs a CUDA device.
`_require_gpu_host()` refuses to run anywhere else with a clear message
rather than a raw import/subprocess traceback, per the "check the device"
rule for this project (dev box is CPU-only; real runs are on the rented
A100/L4 in Colab, per `configs/serve-env.md`). This script is written and
structurally exercised on the dev box (`tests/test_serve_vllm.py` covers the
warmup/sanity math without a live server) and *run* on the GPU host.

Usage, on the GPU host, inside vLLM's own environment:

    python scripts/serve_vllm.py --card A100-SXM4-40GB --precision fp16 --port 8000
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from serving import config, pins, prompts  # noqa: E402

HEALTH_TIMEOUT_S = 300
HEALTH_POLL_S = 2
WARMUP_REQUESTS = 4

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


def _fail(msg: str) -> None:
    print(f"[serve_vllm] refusing to start: {msg}", file=sys.stderr)
    raise SystemExit(1)


def _require_gpu_host() -> None:
    try:
        import torch
    except ImportError:
        _fail("torch is not importable in this environment")
        return
    if sys.platform != "linux":
        _fail(
            f"vLLM has no Windows build -- run this on the Linux GPU host, "
            f"not sys.platform={sys.platform!r}"
        )
    if not torch.cuda.is_available():
        _fail(
            "no CUDA device visible (torch.cuda.is_available() is False) -- "
            "run this on the rented A100/L4, not the CPU dev box"
        )


def launch(card: str, precision: str, port: int) -> subprocess.Popen:
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
    ]
    print(f"[serve_vllm] card={card} precision={precision} revision={revision}")
    print("[serve_vllm] launching:", " ".join(cmd))
    return subprocess.Popen(cmd)


def wait_healthy(port: int, timeout_s: int = HEALTH_TIMEOUT_S) -> None:
    url = f"http://localhost:{port}/health"
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            if requests.get(url, timeout=5).status_code == 200:
                print("[serve_vllm] healthy")
                return
        except requests.RequestException:
            pass
        time.sleep(HEALTH_POLL_S)
    _fail(f"server did not become healthy within {timeout_s}s")


def stream_completion(port: int, prompt: str) -> dict:
    """One streamed completion, client-side timing per §4.1:

        request sent            -> t0
        first streamed chunk    -> t1   TTFT = t1 - t0
        last streamed chunk     -> t2   TPOT = (t2 - t1) / (n_output_tokens - 1)

    `min_tokens` forces the fixed output length (§3) so TPOT is never
    inflated by an early stop -- one framework stopping early would make its
    TPOT look better for free, which is exactly the kind of harness artifact
    §4.1 warns cancels invisibly in review.
    """
    url = f"http://localhost:{port}/v1/completions"
    payload = {
        "model": config.STUDY_MODEL,
        "prompt": prompt,
        "max_tokens": config.OUTPUT_TOKENS,
        "min_tokens": config.OUTPUT_TOKENS,
        "temperature": 0,
        "stream": True,
    }
    t0 = time.monotonic()
    first_chunk_t = None
    last_chunk_t = None
    n_tokens = 0
    with requests.post(url, json=payload, stream=True, timeout=120) as resp:
        resp.raise_for_status()
        for line in resp.iter_lines():
            if not line or not line.startswith(b"data: "):
                continue
            data = line[len(b"data: ") :]
            if data == b"[DONE]":
                break
            now = time.monotonic()
            if first_chunk_t is None:
                first_chunk_t = now
            last_chunk_t = now
            chunk = json.loads(data)
            if chunk["choices"][0].get("text", ""):
                n_tokens += 1

    if first_chunk_t is None:
        raise RuntimeError("no chunks streamed back -- server returned an empty response")

    ttft = first_chunk_t - t0
    tpot = (last_chunk_t - first_chunk_t) / max(n_tokens - 1, 1)
    return {
        "ttft_s": ttft,
        "tpot_s": tpot,
        "n_tokens": n_tokens,
        "wall_s": last_chunk_t - t0,
    }


def warmup(port: int, n: int = WARMUP_REQUESTS) -> None:
    """Fixed count of discarded requests, at concurrency 1 -- §4.3's Day-18
    warmup. The per-concurrency warmup (N discarded requests *at the target
    concurrency*, not concurrency 1) is `sweep.py`'s job on Day 19; this is
    only enough to clear first-request CUDA graph capture / lazy kernel
    compilation before the sanity number is read.
    """
    prompt_set = prompts.load_prompts()
    print(f"[serve_vllm] warmup: {n} discarded requests at concurrency 1")
    for i in range(n):
        stream_completion(port, prompt_set[i % len(prompt_set)])
    print("[serve_vllm] warmup done")


def sanity_check(port: int) -> dict:
    """Day 18's number: tokens/sec at concurrency 1, checked by eye against a
    published figure for this model class (§9) -- 10x off means something is
    wrong today, not on Day 23.
    """
    prompt_set = prompts.load_prompts()
    result = stream_completion(port, prompt_set[0])
    tok_s = config.OUTPUT_TOKENS / result["wall_s"]
    print(
        f"[serve_vllm] concurrency-1 sanity: {tok_s:.1f} tok/s, "
        f"TTFT={result['ttft_s'] * 1000:.0f}ms, TPOT={result['tpot_s'] * 1000:.1f}ms"
    )
    return {**result, "tokens_per_sec": tok_s}


def main() -> None:
    _require_gpu_host()

    ap = argparse.ArgumentParser()
    ap.add_argument("--card", required=True, choices=config.CARDS)
    ap.add_argument("--precision", default="fp16", choices=config.PRECISIONS)
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()

    proc = launch(args.card, args.precision, args.port)
    try:
        wait_healthy(args.port)
        warmup(args.port)
        sanity_check(args.port)
        print("[serve_vllm] server is up -- Ctrl+C to stop")
        proc.wait()
    except KeyboardInterrupt:
        pass
    finally:
        proc.terminate()


if __name__ == "__main__":
    main()
