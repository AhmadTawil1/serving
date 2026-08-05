"""Tests for check_hard_stop.py -- the mechanical Day 4 ("Day 21") decision
rule, exercised against synthetic build-log records rather than a real
Colab run (there is no GPU here to produce a real one).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import check_hard_stop as chs  # noqa: E402
from serving import record  # noqa: E402

A100 = "A100-SXM4-40GB"
L4 = "L4-24GB"


def _rec(card: str, succeeded: bool, precision: str = "fp16") -> dict:
    return {"card": card, "precision": precision, "succeeded": succeeded}


def test_latest_status_per_card_keeps_the_last_record_when_retried():
    records = [
        _rec(A100, succeeded=False),
        _rec(A100, succeeded=True),  # second attempt succeeded
        _rec(L4, succeeded=True),
    ]
    status = chs.latest_status_per_card(records)
    assert status == {A100: True, L4: True}


def test_latest_status_per_card_filters_by_precision():
    records = [
        _rec(A100, succeeded=True, precision="fp16"),
        _rec(A100, succeeded=False, precision="int4"),
    ]
    status = chs.latest_status_per_card(records, precision="fp16")
    assert status == {A100: True}


def test_decide_both_serving():
    result = chs.decide(a100_succeeded=True, l4_succeeded=True)
    assert result["verdict"] == "both_serving"


def test_decide_a100_only():
    result = chs.decide(a100_succeeded=True, l4_succeeded=False)
    assert result["verdict"] == "a100_only"
    assert "portability" in result["action"]


def test_decide_l4_only():
    result = chs.decide(a100_succeeded=False, l4_succeeded=True)
    assert result["verdict"] == "l4_only"


def test_decide_neither():
    result = chs.decide(a100_succeeded=False, l4_succeeded=False)
    assert result["verdict"] == "neither"
    assert "ENTIRELY" in result["action"]


def test_main_reports_neither_when_log_is_empty(tmp_path, monkeypatch, capsys):
    log_path = tmp_path / "empty.jsonl"
    monkeypatch.setattr(sys, "argv", ["check_hard_stop.py", "--log", str(log_path)])
    try:
        chs.main()
        raised = False
    except SystemExit:
        raised = True
    assert raised  # no records at all is a distinct, louder failure than "neither succeeded"


def test_main_end_to_end_both_serving(tmp_path, monkeypatch, capsys):
    log_path = tmp_path / "builds.jsonl"
    record.write_record(log_path, _rec(A100, succeeded=True))
    record.write_record(log_path, _rec(L4, succeeded=True))
    monkeypatch.setattr(sys, "argv", ["check_hard_stop.py", "--log", str(log_path)])

    chs.main()

    out = capsys.readouterr().out
    assert "verdict: both_serving" in out


def test_main_end_to_end_a100_only(tmp_path, monkeypatch, capsys):
    log_path = tmp_path / "builds.jsonl"
    record.write_record(log_path, _rec(A100, succeeded=True))
    record.write_record(log_path, _rec(L4, succeeded=False))
    monkeypatch.setattr(sys, "argv", ["check_hard_stop.py", "--log", str(log_path)])

    chs.main()

    out = capsys.readouterr().out
    assert "verdict: a100_only" in out


def test_main_treats_a_card_never_attempted_as_not_succeeded(tmp_path, monkeypatch, capsys):
    log_path = tmp_path / "builds.jsonl"
    record.write_record(log_path, _rec(A100, succeeded=True))
    # L4 never attempted at all -- must be treated the same as "failed", not
    # crash on a missing key.
    monkeypatch.setattr(sys, "argv", ["check_hard_stop.py", "--log", str(log_path)])

    chs.main()

    out = capsys.readouterr().out
    assert "verdict: a100_only" in out
