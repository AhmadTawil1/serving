"""Closed-loop concurrent load client (§4.2).

`stream_completion` is the streaming-request primitive, shared in spirit
with `serve_vllm.py`'s synchronous version -- both implement the exact same
§4.1 TTFT/TPOT formula, so the Day 18 concurrency-1 sanity number and the
real sweep here are never two slightly-different implementations of "the
same" measurement.

`run_measurement_window` is the closed-loop driver itself: `concurrency`
workers, each holding exactly one request in flight, running for a fixed
wall-clock window. That is what "concurrency N" means throughout this study
(§4.2) -- not a request rate, an in-flight count. `run_warmup` is the same
shape but for a fixed *request count* per worker rather than a time window,
discarded, per §4.3.

Validated against `mock_server.py` before ever touching a real server --
see `tests/test_client.py` and `scripts/validate_client.py` for the §4.2
validity gate this file exists to pass.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import aiohttp
import psutil

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from serving import config, prompts  # noqa: E402

# aiohttp's default TCPConnector caps total connections at 100 -- below the
# top of the concurrency axis (128). `limit=0` means unlimited; the client
# is the thing being proven not to be the bottleneck, so it should not add
# an artificial one of its own.
_CONNECTOR_LIMIT = 0

# How often the CPU sampler wakes up during a measurement window (§4.2).
_CPU_SAMPLE_INTERVAL_S = 0.5


@dataclass
class RequestResult:
    ttft_s: float
    tpot_s: float
    e2e_s: float
    n_tokens: int


@dataclass
class WindowResult:
    request_results: list[RequestResult]
    client_cpu_pct: dict  # {"mean": float, "max": float}
    wall_s: float

    @property
    def n_requests(self) -> int:
        return len(self.request_results)

    @property
    def output_tok_s(self) -> float:
        """Aggregate output-token throughput across every completed request
        in the window -- the raw number the mock-server validity gate
        compares against `mock_server.theoretical_ceiling_tok_s`, and later
        the real sweep's headline numerator (§8).
        """
        total_tokens = sum(r.n_tokens for r in self.request_results)
        return total_tokens / self.wall_s if self.wall_s > 0 else 0.0


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


def _make_session() -> aiohttp.ClientSession:
    connector = aiohttp.TCPConnector(limit=_CONNECTOR_LIMIT)
    return aiohttp.ClientSession(connector=connector)


async def run_warmup(
    base_url: str, concurrency: int, prompts_cycle: list[str], n_per_worker: int
) -> None:
    """Fixed count of discarded requests, at the *target* concurrency (§4.3)
    -- not concurrency 1, and not a fixed number of seconds. Every one of
    `concurrency` workers sends exactly `n_per_worker` requests, sequentially
    within itself, before the measurement window starts.
    """

    async def _one_worker(idx: int, session: aiohttp.ClientSession) -> None:
        for i in range(n_per_worker):
            prompt = prompts_cycle[(idx + i) % len(prompts_cycle)]
            await stream_completion(session, base_url, prompt)

    async with _make_session() as session:
        await asyncio.gather(*[_one_worker(k, session) for k in range(concurrency)])


async def _cpu_sampler(samples: list[float], stop_event: asyncio.Event) -> None:
    proc = psutil.Process()
    proc.cpu_percent(interval=None)  # prime: first call's return value is meaningless
    while not stop_event.is_set():
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=_CPU_SAMPLE_INTERVAL_S)
        except asyncio.TimeoutError:
            pass
        samples.append(proc.cpu_percent(interval=None))


async def run_measurement_window(
    base_url: str, concurrency: int, prompts_cycle: list[str], window_s: float
) -> WindowResult:
    """`concurrency` closed-loop workers running for a fixed wall-clock
    window. In-flight count is exactly `concurrency` throughout -- that is
    what "concurrency N" means in this study (§4.2), not a request rate. A
    request already in flight when the window ends is allowed to finish
    naturally (no new ones start after the deadline); this is the standard
    windowed-measurement tradeoff and is why `n_requests` varies slightly
    run to run rather than being fixed in advance.

    Client CPU is sampled throughout (§4.2) -- a pegged client invalidates
    the point at this concurrency, and the record schema (§8) carries
    `client_cpu_pct` precisely so that can be checked later, not just at
    validation time.
    """
    results: list[RequestResult] = []
    cpu_samples: list[float] = []
    stop_event = asyncio.Event()

    t_start = time.monotonic()
    deadline = t_start + window_s

    async def _one_worker(idx: int, session: aiohttp.ClientSession) -> None:
        i = 0
        while time.monotonic() < deadline:
            prompt = prompts_cycle[(idx + i) % len(prompts_cycle)]
            r = await stream_completion(session, base_url, prompt)
            results.append(r)
            i += 1

    sampler_task = asyncio.create_task(_cpu_sampler(cpu_samples, stop_event))
    async with _make_session() as session:
        await asyncio.gather(*[_one_worker(k, session) for k in range(concurrency)])
    stop_event.set()
    await sampler_task

    wall_s = time.monotonic() - t_start
    cpu_pct = {
        "mean": sum(cpu_samples) / len(cpu_samples) if cpu_samples else 0.0,
        "max": max(cpu_samples) if cpu_samples else 0.0,
    }
    return WindowResult(request_results=results, client_cpu_pct=cpu_pct, wall_s=wall_s)


async def _main_async(args: argparse.Namespace) -> None:
    prompt_set = prompts.load_prompts()
    print(f"[client] warmup: {args.warmup_requests} req/worker @ concurrency {args.concurrency}")
    await run_warmup(args.base_url, args.concurrency, prompt_set, args.warmup_requests)
    print(f"[client] measuring for {args.window_s}s @ concurrency {args.concurrency}")
    window = await run_measurement_window(args.base_url, args.concurrency, prompt_set, args.window_s)
    print(
        f"[client] n_requests={window.n_requests} wall_s={window.wall_s:.2f} "
        f"output_tok_s={window.output_tok_s:.1f} "
        f"client_cpu_pct(mean/max)={window.client_cpu_pct['mean']:.1f}/{window.client_cpu_pct['max']:.1f}"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Ad hoc single-concurrency-level check")
    ap.add_argument("--base-url", default="http://localhost:8000")
    ap.add_argument("--concurrency", type=int, default=1)
    ap.add_argument("--window-s", type=float, default=10.0)
    ap.add_argument("--warmup-requests", type=int, default=2)
    args = ap.parse_args()
    asyncio.run(_main_async(args))


if __name__ == "__main__":
    main()
