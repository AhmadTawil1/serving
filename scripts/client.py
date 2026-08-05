"""Closed-loop concurrent load client (§4.2).

**Skeleton as of Day 18.** The full N-worker closed-loop driver, per-run
aggregation, and client CPU-utilisation sampling are Day 19 work, built and
validated against `mock_server.py` before they touch a real server (§4.2's
validity gate). What exists today is the streaming-request primitive and its
TTFT/TPOT computation -- the async twin of `serve_vllm.py`'s
`stream_completion`, sharing the exact same §4.1 formula so the concurrency-1
sanity number (Day 18, synchronous `requests`) and the real sweep (Day 19+,
`aiohttp`) are never two slightly-different implementations of "the same"
measurement. That drift is exactly the kind of harness artifact this project
is designed to not produce twice.
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import aiohttp

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from serving import config  # noqa: E402


@dataclass
class RequestResult:
    ttft_s: float
    tpot_s: float
    e2e_s: float
    n_tokens: int


async def stream_completion(
    session: aiohttp.ClientSession, base_url: str, prompt: str
) -> RequestResult:
    """One streamed request against the OpenAI-compatible endpoint.

        request sent            -> t0
        first streamed chunk    -> t1        TTFT = t1 - t0
        last streamed chunk     -> t2        TPOT = (t2 - t1) / (n_output_tokens - 1)

    `min_tokens` forces the fixed output length (§3) so a framework that
    stops early can't look faster for free. Uses `aiohttp`, not `requests`
    -- this is the transport that has to hold 128 concurrent in-flight
    streams without becoming the bottleneck itself (§4.2), which a
    thread-per-request `requests` client cannot be trusted to do cleanly.
    """
    payload = {
        "model": config.STUDY_MODEL,
        "prompt": prompt,
        "max_tokens": config.OUTPUT_TOKENS,
        "min_tokens": config.OUTPUT_TOKENS,
        "temperature": 0,
        "stream": True,
    }
    t0 = time.monotonic()
    first_chunk_t: float | None = None
    last_chunk_t: float | None = None
    n_tokens = 0

    async with session.post(f"{base_url}/v1/completions", json=payload) as resp:
        resp.raise_for_status()
        async for raw_line in resp.content:
            line = raw_line.strip()
            if not line.startswith(b"data: "):
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

    if first_chunk_t is None or last_chunk_t is None:
        raise RuntimeError("no chunks streamed back -- server returned an empty response")

    ttft = first_chunk_t - t0
    tpot = (last_chunk_t - first_chunk_t) / max(n_tokens - 1, 1)
    return RequestResult(
        ttft_s=ttft, tpot_s=tpot, e2e_s=last_chunk_t - t0, n_tokens=n_tokens
    )


async def worker(
    base_url: str,
    prompts_cycle: list[str],
    n_requests: int,
    results: list[RequestResult],
) -> None:
    """One closed-loop worker: send, await the full response, send the next.
    In-flight count for this worker is always exactly 1; N of these running
    concurrently *is* the concurrency-N definition (§4.2).

    TODO Day 19: wire into `sweep.py` -- fixed warmup count at the *target*
    concurrency (discarded, §4.3), then a fixed measurement window/count,
    with per-repeat records written via `serving/record.py` (not yet
    written).
    """
    raise NotImplementedError("Day 19: sweep.py's closed-loop driver")


async def run_concurrency_level(
    base_url: str, concurrency: int, n_requests_per_worker: int
) -> list[RequestResult]:
    """Launch `concurrency` workers, collect their RequestResults, and sample
    client-side CPU utilisation throughout (§4.2 -- a pegged client CPU
    invalidates the point).

    TODO Day 19.
    """
    raise NotImplementedError("Day 19: sweep.py's closed-loop driver")


if __name__ == "__main__":
    raise SystemExit(
        "client.py is a skeleton as of Day 18. For the concurrency-1 sanity "
        "number, run scripts/serve_vllm.py on the GPU host. The concurrency "
        "sweep driver (mock_server.py validation, N-worker closed loop, "
        "record.py output) lands Day 19."
    )
