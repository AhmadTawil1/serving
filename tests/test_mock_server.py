"""Tests for mock_server.py -- the §4.2 validity-gate target. Runs entirely
in-process via aiohttp's TestServer/TestClient, no real socket, no GPU.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest
from aiohttp.test_utils import TestClient, TestServer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import mock_server  # noqa: E402


def _run(coro):
    return asyncio.run(coro)


def test_health_returns_ok():
    async def _t():
        async with TestClient(TestServer(mock_server.create_app())) as client:
            resp = await client.get("/health")
            assert resp.status == 200
            assert (await resp.json())["status"] == "ok"

    _run(_t())


def test_completions_streams_exactly_max_tokens_chunks_then_done():
    async def _t():
        async with TestClient(TestServer(mock_server.create_app())) as client:
            resp = await client.post(
                "/v1/completions",
                json={
                    "model": "x",
                    "prompt": "hi",
                    "max_tokens": 5,
                    "min_tokens": 5,
                    "temperature": 0,
                    "stream": True,
                },
            )
            assert resp.status == 200
            lines = [ln.strip() async for ln in resp.content if ln.strip()]
            data_lines = [ln for ln in lines if ln.startswith(b"data: ") and ln != b"data: [DONE]"]
            assert len(data_lines) == 5
            assert lines[-1] == b"data: [DONE]"

    _run(_t())


def test_completions_honours_whichever_of_max_min_tokens_is_larger():
    # client.py always sends max_tokens == min_tokens == OUTPUT_TOKENS, but the
    # server should do something sane (emit the larger count) if they differ,
    # rather than silently truncating to the forced minimum.
    async def _t():
        async with TestClient(TestServer(mock_server.create_app())) as client:
            resp = await client.post(
                "/v1/completions",
                json={
                    "model": "x",
                    "prompt": "hi",
                    "max_tokens": 3,
                    "min_tokens": 7,
                    "temperature": 0,
                    "stream": True,
                },
            )
            lines = [ln.strip() async for ln in resp.content if ln.strip()]
            data_lines = [ln for ln in lines if ln.startswith(b"data: ") and ln != b"data: [DONE]"]
            assert len(data_lines) == 7

    _run(_t())


def test_theoretical_ceiling_scales_linearly_with_concurrency():
    c1 = mock_server.theoretical_ceiling_tok_s(1, output_tokens=128)
    c2 = mock_server.theoretical_ceiling_tok_s(2, output_tokens=128)
    assert c2 == pytest.approx(2 * c1)


def test_theoretical_ceiling_matches_the_fixed_delays_directly():
    wall = mock_server.TTFT_DELAY_S + 9 * mock_server.TOKEN_SLEEP_S
    expected = 10 / wall
    assert mock_server.theoretical_ceiling_tok_s(1, output_tokens=10) == pytest.approx(expected)
