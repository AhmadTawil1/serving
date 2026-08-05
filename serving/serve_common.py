"""Shared server-lifecycle helpers for the framework-launcher scripts
(`scripts/serve_vllm.py`, `scripts/serve_trtllm.py`).

Both frameworks are served behind an identical OpenAI-compatible completions
endpoint (§3.1 of `PAPER.md`), so health-check polling, warmup, the
concurrency-1 sanity number, and the TTFT/TPOT formula itself belong in
exactly one place. REFERENCE.md's ground rule 3 ("one harness, three
studies") applies just as much within this one measurement, across its two
frameworks, as it does across the programme's three measurements -- a
`stream_completion` reimplemented a third time here (after `client.py`'s
async version) is exactly the kind of drift §4.1 warns produces a "result"
that is really a harness artifact.
"""

from __future__ import annotations

import json
import sys
import time

import requests

from . import config, prompts

HEALTH_TIMEOUT_S = 300
HEALTH_POLL_S = 2
WARMUP_REQUESTS = 4


def fail(label: str, msg: str) -> None:
    print(f"[{label}] refusing to start: {msg}", file=sys.stderr)
    raise SystemExit(1)


def require_gpu_host(label: str) -> None:
    """Both vLLM and TensorRT-LLM are Linux + CUDA only (confirmed 5 Aug
    2026, LOG.md Day 1 -- neither publishes a Windows wheel at any version).
    Refuse to run anywhere else with a clear message rather than a raw
    import/subprocess traceback; the dev box is CPU-only by design, and real
    runs are on the rented A100/L4 in Colab (`configs/serve-env.md`).
    """
    try:
        import torch
    except ImportError:
        fail(label, "torch is not importable in this environment")
        return
    if sys.platform != "linux":
        fail(
            label,
            f"no Windows build for this framework -- run on the Linux GPU "
            f"host, not sys.platform={sys.platform!r}",
        )
    if not torch.cuda.is_available():
        fail(
            label,
            "no CUDA device visible (torch.cuda.is_available() is False) -- "
            "run this on the rented A100/L4, not the CPU dev box",
        )


def wait_healthy(
    label: str, port: int, health_path: str = "/health", timeout_s: int = HEALTH_TIMEOUT_S
) -> None:
    url = f"http://localhost:{port}{health_path}"
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            if requests.get(url, timeout=5).status_code == 200:
                print(f"[{label}] healthy")
                return
        except requests.RequestException:
            pass
        time.sleep(HEALTH_POLL_S)
    fail(label, f"server did not become healthy within {timeout_s}s")


def stream_completion(port: int, prompt: str) -> dict:
    """One streamed completion against the OpenAI-compatible endpoint,
    client-side timing per §4.1:

        request sent            -> t0
        first streamed chunk    -> t1   TTFT = t1 - t0
        last streamed chunk     -> t2   TPOT = (t2 - t1) / (n_output_tokens - 1)

    `min_tokens` forces the fixed output length (§3) so TPOT is never
    inflated by an early stop. This is the *synchronous* twin of
    `scripts/client.py`'s async `stream_completion` -- same formula, same
    payload shape, so a one-off sanity check here and the real concurrency
    sweep there can never quietly become two different measurements of "the
    same" thing.
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


def warmup(label: str, port: int, n: int = WARMUP_REQUESTS) -> None:
    """Fixed count of discarded requests, at concurrency 1 -- enough to clear
    first-request engine artifacts (CUDA graph capture, lazy kernel
    compilation) before the sanity number is read. The per-*concurrency*
    warmup (§4.3, distributed across N workers) is `sweep.py`'s job.
    """
    prompt_set = prompts.load_prompts()
    print(f"[{label}] warmup: {n} discarded requests at concurrency 1")
    for i in range(n):
        stream_completion(port, prompt_set[i % len(prompt_set)])
    print(f"[{label}] warmup done")


def sanity_check(label: str, port: int) -> dict:
    """Tokens/sec at concurrency 1, checked by eye against a published
    figure for this model class -- 10x off means something is wrong today,
    not discovered three days later.
    """
    prompt_set = prompts.load_prompts()
    result = stream_completion(port, prompt_set[0])
    tok_s = config.OUTPUT_TOKENS / result["wall_s"]
    print(
        f"[{label}] concurrency-1 sanity: {tok_s:.1f} tok/s, "
        f"TTFT={result['ttft_s'] * 1000:.0f}ms, TPOT={result['tpot_s'] * 1000:.1f}ms"
    )
    return {**result, "tokens_per_sec": tok_s}
