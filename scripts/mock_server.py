"""Mock OpenAI-compatible completions server -- the validity gate in §4.2.

Returns a canned token stream on a fixed, known sleep. Not a real model,
just enough of the wire format for `client.py` to drive it exactly as it
would drive vLLM/TensorRT-LLM. Its purpose is to answer one question before
any real number gets trusted: if the client can't sustain concurrency 128
against a server whose own ceiling is *known in advance*, the client itself
is the bottleneck, and the top of the concurrency axis isn't measurable yet.

Deliberately the one piece of the Day 19 harness that's fully runnable and
checkable on the CPU dev box -- no GPU, no model weights, no vLLM. `asyncio
.sleep` never blocks the event loop, so this server's own capacity is
effectively unlimited; any shortfall between achieved and theoretical
throughput is attributable to the client, which is the entire point.

**Windows-specific fix, found while building this (LOG.md, Day 2):** the
default `ProactorEventLoop` silently collapses `asyncio.sleep(0.01)` calls
that are interleaved with `aiohttp` `StreamResponse.write()`s -- a 10-call
loop of 10ms sleeps completed in ~0ms instead of ~100ms, reproduced in a
minimal handler with no other code involved, and confirmed fixed by
switching to `WindowsSelectorEventLoopPolicy`. Plain `asyncio.sleep()` in a
loop with no aiohttp writes involved was unaffected, so this is specifically
a Proactor + stream-write interaction, not a general timer bug. Applied only
on `win32`; Colab (Linux) never hits this and keeps the default loop.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from aiohttp import web

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from serving import config  # noqa: E402

# Fixed, known delays. A real server's TTFT/TPOT are opaque; this one's are
# the two numbers below, so the "theoretical ceiling" a sweep is checked
# against is computed directly from them, not guessed or curve-fit.
TTFT_DELAY_S = 0.05
TOKEN_SLEEP_S = 0.01

TOKEN_TEXT_KEY: web.AppKey[str] = web.AppKey("token_text", str)


def theoretical_ceiling_tok_s(
    concurrency: int, output_tokens: int = config.OUTPUT_TOKENS
) -> float:
    """Aggregate tokens/sec an unbottlenecked client should achieve against
    this server at `concurrency`, given its fixed per-request timing.
    """
    per_request_wall_s = TTFT_DELAY_S + (output_tokens - 1) * TOKEN_SLEEP_S
    return concurrency * output_tokens / per_request_wall_s


async def health(request: web.Request) -> web.Response:
    return web.json_response({"status": "ok"})


async def completions(request: web.Request) -> web.StreamResponse:
    payload = await request.json()
    max_tokens = payload.get("max_tokens", config.OUTPUT_TOKENS)
    min_tokens = payload.get("min_tokens", max_tokens)
    # A real server forced to a fixed length via min_tokens emits exactly
    # that many; the canned stream mirrors that rather than always using
    # max_tokens, so a caller that forgets min_tokens is caught here too.
    n_tokens = max(max_tokens, min_tokens)
    token_text = request.app[TOKEN_TEXT_KEY]

    resp = web.StreamResponse(status=200, headers={"Content-Type": "text/event-stream"})
    await resp.prepare(request)

    await asyncio.sleep(TTFT_DELAY_S)
    for i in range(n_tokens):
        chunk = {"choices": [{"text": token_text}]}
        await resp.write(f"data: {json.dumps(chunk)}\n\n".encode())
        if i < n_tokens - 1:
            await asyncio.sleep(TOKEN_SLEEP_S)
    await resp.write(b"data: [DONE]\n\n")
    await resp.write_eof()
    return resp


def create_app(token_text: str = "x") -> web.Application:
    """`token_text` is the only thing distinguishing one mock server
    instance from another -- used by `tests/test_check_parity.py` to stand
    up two instances that disagree, so the parity check's failure path is
    tested against a real (if fake) divergence, not just mocked out.
    """
    app = web.Application()
    app[TOKEN_TEXT_KEY] = token_text
    app.router.add_get("/health", health)
    app.router.add_post("/v1/completions", completions)
    return app


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--token-text", default="x")
    args = ap.parse_args()
    web.run_app(create_app(token_text=args.token_text), port=args.port)


if __name__ == "__main__":
    main()
