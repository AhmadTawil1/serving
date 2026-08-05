"""Tests for record.py's schema assembly (§8)."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from serving import config, record


@dataclass
class _FakeResult:
    ttft_s: float
    tpot_s: float
    e2e_s: float
    n_tokens: int


def test_config_id_is_order_independent():
    assert record.config_id({"x": 1, "y": 2}) == record.config_id({"y": 2, "x": 1})


def test_config_id_differs_for_different_configs():
    assert record.config_id({"x": 1}) != record.config_id({"x": 2})


def test_percentile_of_empty_raises():
    with pytest.raises(ValueError):
        record.percentile([], 50)


def test_build_record_raises_on_no_results():
    with pytest.raises(ValueError, match="no request results"):
        record.build_record(
            cfg={}, request_results=[], repeat=1, of_repeats=1,
            warmup_requests=0, window_s=1.0, prov={},
        )


def test_build_record_computes_throughput_and_percentiles():
    results = [
        _FakeResult(ttft_s=0.1, tpot_s=0.01, e2e_s=1.0, n_tokens=10),
        _FakeResult(ttft_s=0.2, tpot_s=0.02, e2e_s=2.0, n_tokens=10),
    ]
    rec = record.build_record(
        cfg={"framework": "mock"},
        request_results=results,
        repeat=1,
        of_repeats=3,
        warmup_requests=4,
        window_s=2.0,
        prov={"gpu": "none"},
    )
    assert rec["throughput"]["output_tok_s"] == pytest.approx(20 / 2.0)
    assert rec["latency"]["ttft_ms"]["p50"] == pytest.approx(150.0)  # median of 100ms, 200ms
    assert rec["run"]["n_requests"] == 2
    assert rec["run"]["repeat"] == 1
    assert rec["run"]["of_repeats"] == 3
    assert rec["prov"] == {"gpu": "none"}
    assert rec["config_id"] == record.config_id({"framework": "mock"})


def test_build_record_defaults_unset_blocks_to_schema_shape():
    results = [_FakeResult(ttft_s=0.1, tpot_s=0.01, e2e_s=1.0, n_tokens=10)]
    rec = record.build_record(
        cfg={}, request_results=results, repeat=1, of_repeats=1,
        warmup_requests=0, window_s=1.0, prov={},
    )
    assert set(rec) == {
        "config", "config_id", "throughput", "latency", "cost",
        "resources", "build", "quality", "run", "prov",
    }
    assert rec["quality"] == {
        "checked": False, "reference": None, "exact_match": None, "passed": None,
    }
    assert rec["build"] == {"engine_build_s": 0, "engine_id": None}


def test_build_record_goodput_counts_requests_meeting_slo(monkeypatch):
    monkeypatch.setattr(config, "SLO_TTFT_MS", 150)
    monkeypatch.setattr(config, "SLO_TPOT_MS", 50)
    results = [
        _FakeResult(ttft_s=0.1, tpot_s=0.01, e2e_s=1.0, n_tokens=10),  # meets both
        _FakeResult(ttft_s=0.2, tpot_s=0.01, e2e_s=1.0, n_tokens=10),  # TTFT over threshold
    ]
    rec = record.build_record(
        cfg={}, request_results=results, repeat=1, of_repeats=1,
        warmup_requests=0, window_s=2.0, prov={},
    )
    assert rec["throughput"]["goodput_req_s"] == pytest.approx(1 / 2.0)


def test_write_record_and_read_records_roundtrip(tmp_path):
    path = tmp_path / "out.jsonl"
    rec1, rec2 = {"a": 1}, {"a": 2}

    record.write_record(path, rec1)
    record.write_record(path, rec2)

    assert record.read_records(path) == [rec1, rec2]


def test_write_record_appends_rather_than_overwrites(tmp_path):
    path = tmp_path / "out.jsonl"
    record.write_record(path, {"a": 1})
    before = path.read_text()
    record.write_record(path, {"a": 2})
    assert path.read_text().startswith(before)


def test_read_records_missing_file_returns_empty(tmp_path):
    assert record.read_records(tmp_path / "nope.jsonl") == []
