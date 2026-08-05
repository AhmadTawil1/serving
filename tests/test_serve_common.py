"""Tests for serve_common.py -- shared server-lifecycle helpers used by
serve_vllm.py and serve_trtllm.py.

Runs `mock_server.py` as a **real subprocess**, not an in-process
`aiohttp.test_utils.TestServer` -- serve_common's functions are plain
synchronous `requests` calls with no event loop of their own, and mixing
them via `asyncio.to_thread` into the *same* process/event loop as an
in-process test server produced a reproducible ~2s stall on this Windows
box (a `requests`+active-Selector-loop contention artifact, confirmed by
timing a plain `requests` call with no event loop involved at all: 0.078s,
vs. ~2.1s through `to_thread` against an in-process server). A real
subprocess is also what these functions actually talk to in practice
(`serve_vllm.py`/`serve_trtllm.py` always drive a separate process), so this
is the more representative test setup, not just the workaround.
"""

from __future__ import annotations

import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest
import requests

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import mock_server  # noqa: E402
from serving import config, serve_common  # noqa: E402


def _free_port() -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


@pytest.fixture(scope="module", autouse=True)
def fast_output_tokens_module():
    # module-scoped monkeypatch (no `monkeypatch` fixture at module scope) --
    # restored manually at the end.
    original = config.OUTPUT_TOKENS
    config.OUTPUT_TOKENS = 4
    yield
    config.OUTPUT_TOKENS = original


@pytest.fixture(scope="module")
def mock_server_port():
    port = _free_port()
    proc = subprocess.Popen(
        [sys.executable, "scripts/mock_server.py", "--port", str(port)],
        cwd=REPO_ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    deadline = time.monotonic() + 15
    healthy = False
    while time.monotonic() < deadline:
        try:
            if requests.get(f"http://localhost:{port}/health", timeout=1).status_code == 200:
                healthy = True
                break
        except requests.RequestException:
            pass
        time.sleep(0.2)
    if not healthy:
        proc.terminate()
        pytest.fail("mock_server.py subprocess never became healthy")

    yield port

    proc.terminate()
    proc.wait(timeout=5)


def test_fail_prints_label_and_reason_then_raises_systemexit(capsys):
    with pytest.raises(SystemExit):
        serve_common.fail("test-label", "some reason")
    captured = capsys.readouterr()
    assert "[test-label]" in captured.err
    assert "some reason" in captured.err


def test_require_gpu_host_fails_on_this_cpu_dev_box():
    # This suite runs on the CPU dev box -- confirming the guard fires here
    # is exactly the environment it exists to refuse.
    with pytest.raises(SystemExit):
        serve_common.require_gpu_host("test-label")


def test_wait_healthy_succeeds_against_a_live_server(mock_server_port):
    serve_common.wait_healthy("test", mock_server_port, "/health", timeout_s=5)


def test_wait_healthy_times_out_against_nothing_listening(monkeypatch):
    monkeypatch.setattr(serve_common, "HEALTH_POLL_S", 0.05)  # keep the test fast
    unused_port = _free_port()  # bound-then-closed: reliably nothing listening
    with pytest.raises(SystemExit):
        serve_common.wait_healthy("test", unused_port, "/health", timeout_s=0.3)


def test_stream_completion_reports_correct_ttft_tpot_and_token_count(mock_server_port):
    result = serve_common.stream_completion(mock_server_port, "hello")
    assert result["n_tokens"] == 4
    assert result["ttft_s"] == pytest.approx(mock_server.TTFT_DELAY_S, abs=0.05)
    assert result["tpot_s"] == pytest.approx(mock_server.TOKEN_SLEEP_S, abs=0.02)


def test_warmup_sends_exactly_n_requests(mock_server_port):
    calls = []
    original = serve_common.stream_completion

    def counting(port_, prompt):
        calls.append(prompt)
        return original(port_, prompt)

    import unittest.mock as um

    with um.patch.object(serve_common, "stream_completion", counting):
        serve_common.warmup("test", mock_server_port, 3)

    assert len(calls) == 3


def test_sanity_check_returns_a_positive_tokens_per_sec(mock_server_port):
    result = serve_common.sanity_check("test", mock_server_port)
    assert result["tokens_per_sec"] > 0
