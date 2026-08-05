"""Integration tests for client.py's closed-loop driver, against
mock_server.py in-process (aiohttp TestServer) -- no real socket beyond
loopback, no GPU. `config.OUTPUT_TOKENS` is monkeypatched down for most of
these so the suite doesn't spend real wall-clock time on the full 128-token
default (§4.1's formulas don't care about the token count, only that it's
consistent between client and server).
"""

from __future__ import annotations

import asyncio
import sys
import unittest.mock
from pathlib import Path

import aiohttp
import pytest
from aiohttp.test_utils import TestServer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import client  # noqa: E402
import mock_server  # noqa: E402


def _run(coro):
    return asyncio.run(coro)


class _MockServerCtx:
    """Starts mock_server.py's app on an ephemeral port for one test."""

    async def __aenter__(self) -> str:
        self._server = TestServer(mock_server.create_app())
        await self._server.start_server()
        return str(self._server.make_url(""))

    async def __aexit__(self, *exc):
        await self._server.close()


@pytest.fixture(autouse=True)
def fast_output_tokens(monkeypatch):
    """4 tokens instead of 128 -- keeps every test in this file fast while
    still exercising the real TTFT/TPOT/count logic end to end.
    """
    monkeypatch.setattr(client.config, "OUTPUT_TOKENS", 4)


def test_stream_completion_reports_correct_ttft_tpot_and_token_count():
    async def _t():
        async with _MockServerCtx() as base_url:
            async with aiohttp.ClientSession() as session:
                result = await client.stream_completion(session, base_url, "hello")
            assert result.n_tokens == 4
            assert result.ttft_s == pytest.approx(mock_server.TTFT_DELAY_S, abs=0.05)
            assert result.tpot_s == pytest.approx(mock_server.TOKEN_SLEEP_S, abs=0.02)

    _run(_t())


def test_run_warmup_sends_exactly_n_per_worker_requests():
    async def _t():
        async with _MockServerCtx() as base_url:
            call_count = {"n": 0}
            original = client.stream_completion

            async def counting(session, url, prompt):
                call_count["n"] += 1
                return await original(session, url, prompt)

            with unittest.mock.patch.object(client, "stream_completion", counting):
                await client.run_warmup(base_url, concurrency=3, prompts_cycle=["a", "b"], n_per_worker=2)

            assert call_count["n"] == 6  # 3 workers x 2 requests each

    _run(_t())


def test_run_measurement_window_holds_exact_concurrency_in_flight():
    async def _t():
        async with _MockServerCtx() as base_url:
            # Window (30ms) comfortably larger than asyncio task-startup
            # jitter but still shorter than one full request (~60-80ms at 4
            # tokens), so every one of the `concurrency` workers gets to
            # start exactly one request before the deadline, and none
            # starts a second -- a direct check that in-flight count equals
            # `concurrency`, not more or fewer. (An earlier version used a
            # 10ms window and flaked under scheduler jitter: some workers'
            # first deadline check landed *after* asyncio.gather had already
            # spent the whole budget just starting tasks, so they logged 0
            # requests. Not a bug in the client -- a too-tight test.)
            window = await client.run_measurement_window(
                base_url, concurrency=5, prompts_cycle=["a"], window_s=0.03
            )
            assert window.n_requests == 5
            assert all(r.n_tokens == 4 for r in window.request_results)

    _run(_t())


def test_run_measurement_window_output_tok_s_matches_manual_computation():
    async def _t():
        async with _MockServerCtx() as base_url:
            window = await client.run_measurement_window(
                base_url, concurrency=2, prompts_cycle=["a"], window_s=0.01
            )
            total_tokens = sum(r.n_tokens for r in window.request_results)
            assert window.output_tok_s == pytest.approx(total_tokens / window.wall_s)

    _run(_t())


def test_run_measurement_window_samples_client_cpu():
    async def _t():
        async with _MockServerCtx() as base_url:
            window = await client.run_measurement_window(
                base_url, concurrency=2, prompts_cycle=["a"], window_s=0.3
            )
            assert "mean" in window.client_cpu_pct
            assert "max" in window.client_cpu_pct
            assert window.client_cpu_pct["max"] >= window.client_cpu_pct["mean"]

    _run(_t())
