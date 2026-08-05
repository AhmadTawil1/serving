"""Tests for check_parity.py, against two real mock_server.py subprocesses
-- one the "vLLM" stand-in, one the "TensorRT-LLM" stand-in, distinguished
by `--token-text` so the divergence-detection path is exercised against a
real (if fake) mismatch, not mocked out. Same subprocess-based approach as
`test_serve_common.py`, for the same reason: these are plain synchronous
`requests` calls, best tested against a genuinely separate process rather
than sharing an event loop with an in-process test server.
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

from serving import config  # noqa: E402
import check_parity  # noqa: E402


def _free_port() -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _start_mock_server(port: int, token_text: str) -> subprocess.Popen:
    proc = subprocess.Popen(
        [sys.executable, "scripts/mock_server.py", "--port", str(port), "--token-text", token_text],
        cwd=REPO_ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        try:
            if requests.get(f"http://localhost:{port}/health", timeout=1).status_code == 200:
                return proc
        except requests.RequestException:
            pass
        time.sleep(0.2)
    proc.terminate()
    pytest.fail(f"mock_server.py on port {port} never became healthy")


@pytest.fixture(scope="module", autouse=True)
def fast_output_tokens_module():
    original = config.OUTPUT_TOKENS
    config.OUTPUT_TOKENS = 4
    yield
    config.OUTPUT_TOKENS = original


@pytest.fixture
def two_identical_servers():
    port_a, port_b = _free_port(), _free_port()
    proc_a = _start_mock_server(port_a, "x")
    proc_b = _start_mock_server(port_b, "x")
    yield port_a, port_b
    proc_a.terminate()
    proc_b.terminate()


@pytest.fixture
def two_divergent_servers():
    port_a, port_b = _free_port(), _free_port()
    proc_a = _start_mock_server(port_a, "x")
    proc_b = _start_mock_server(port_b, "y")  # different "model" output
    yield port_a, port_b
    proc_a.terminate()
    proc_b.terminate()


def test_fetch_completion_text_returns_the_concatenated_tokens(two_identical_servers):
    port_a, _ = two_identical_servers
    text = check_parity.fetch_completion_text(port_a, "hello")
    assert text == "xxxx"  # 4 tokens (fast_output_tokens_module), each "x"


def test_check_passes_when_both_servers_agree(two_identical_servers):
    port_a, port_b = two_identical_servers
    result = check_parity.check(port_a, port_b, "vllm", "tensorrt-llm")
    assert result["exact_match"] == 1.0
    assert result["passed"] is True


def test_check_fails_when_servers_diverge(two_divergent_servers):
    port_a, port_b = two_divergent_servers
    result = check_parity.check(port_a, port_b, "vllm", "tensorrt-llm")
    assert result["exact_match"] == 0.0
    assert result["passed"] is False


def test_main_exits_nonzero_on_divergence(two_divergent_servers, monkeypatch):
    port_a, port_b = two_divergent_servers
    monkeypatch.setattr(
        sys, "argv",
        ["check_parity.py", "--port-a", str(port_a), "--port-b", str(port_b)],
    )
    with pytest.raises(SystemExit) as exc_info:
        check_parity.main()
    assert exc_info.value.code != 0


def test_main_exits_zero_on_agreement(two_identical_servers, monkeypatch):
    port_a, port_b = two_identical_servers
    monkeypatch.setattr(
        sys, "argv",
        ["check_parity.py", "--port-a", str(port_a), "--port-b", str(port_b)],
    )
    check_parity.main()  # should not raise
