"""Tests for build_log.py's timing/RAM/VRAM sampling and record assembly."""

from __future__ import annotations

import sys

from serving import build_log


def test_run_timed_with_peaks_captures_wall_time_and_output():
    result = build_log.run_timed_with_peaks(
        [sys.executable, "-c", "print('hello'); import time; time.sleep(0.3)"]
    )
    assert result["returncode"] == 0
    assert "hello" in result["stdout"]
    assert result["wall_s"] >= 0.25
    assert result["peak_host_ram_mb"] is not None
    assert result["peak_host_ram_mb"] > 0


def test_run_timed_with_peaks_captures_nonzero_returncode_and_stderr():
    result = build_log.run_timed_with_peaks(
        [sys.executable, "-c", "import sys; sys.stderr.write('boom'); sys.exit(3)"]
    )
    assert result["returncode"] == 3
    assert "boom" in result["stderr"]


def test_peak_vram_mb_is_none_when_nvidia_smi_is_unavailable(monkeypatch):
    monkeypatch.setattr(build_log, "_nvidia_smi_used_mb", lambda: None)
    result = build_log.run_timed_with_peaks([sys.executable, "-c", "pass"])
    assert result["peak_vram_mb"] is None


def test_nvidia_smi_used_mb_returns_none_on_this_gpu_less_dev_box():
    # This suite runs on the CPU dev box -- nvidia-smi genuinely isn't
    # there, so this exercises the real except-and-return-None path rather
    # than a mocked one.
    assert build_log._nvidia_smi_used_mb() is None


def test_build_record_sums_wall_time_and_takes_the_max_of_peaks():
    steps = [
        {"wall_s": 1.0, "peak_host_ram_mb": 100, "peak_vram_mb": None},
        {"wall_s": 2.0, "peak_host_ram_mb": 150, "peak_vram_mb": 5000},
    ]
    rec = build_log.build_record(
        card="A100-SXM4-40GB",
        precision="fp16",
        framework="tensorrt-llm",
        engine_id="eng-1",
        step_results=steps,
        succeeded=True,
    )
    assert rec["total_wall_s"] == 3.0
    assert rec["peak_host_ram_mb"] == 150
    assert rec["peak_vram_mb"] == 5000
    assert rec["succeeded"] is True
    assert rec["steps"] == steps


def test_build_record_handles_all_none_peaks():
    steps = [{"wall_s": 1.0, "peak_host_ram_mb": None, "peak_vram_mb": None}]
    rec = build_log.build_record(
        card="L4-24GB",
        precision="fp16",
        framework="tensorrt-llm",
        engine_id=None,
        step_results=steps,
        succeeded=False,
    )
    assert rec["peak_host_ram_mb"] is None
    assert rec["peak_vram_mb"] is None
    assert rec["succeeded"] is False


def test_write_build_record_appends_jsonl(tmp_path):
    path = tmp_path / "builds.jsonl"
    build_log.write_build_record(path, {"a": 1})
    build_log.write_build_record(path, {"a": 2})
    lines = path.read_text().splitlines()
    assert len(lines) == 2
