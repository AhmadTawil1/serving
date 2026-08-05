"""Unit tests for the per-precision quality gate (§4.4)."""

from __future__ import annotations

import json
import math

import pytest

from serving import quality


@pytest.fixture(autouse=True)
def reset_threshold(monkeypatch):
    monkeypatch.setattr(quality, "EXACT_MATCH_THRESHOLD", None)


def test_exact_match_rate_all_match():
    assert quality.exact_match_rate(["a", "b", "c"], ["a", "b", "c"]) == 1.0


def test_exact_match_rate_partial_match_strips_whitespace():
    # "b " vs "b" should still count as a match -- trailing whitespace from a
    # streamed response is a transport artifact, not a quality difference.
    assert quality.exact_match_rate(["a", "b ", "x"], ["a", "b", "c"]) == 2 / 3


def test_exact_match_rate_rejects_length_mismatch():
    with pytest.raises(ValueError, match="count mismatch"):
        quality.exact_match_rate(["a"], ["a", "b"])


def test_exact_match_rate_rejects_empty_input():
    with pytest.raises(ValueError, match="no outputs"):
        quality.exact_match_rate([], [])


def test_perplexity_of_confident_correct_prediction_is_near_one():
    # sum_logprob close to 0 (i.e. log(1.0) per token) -> perplexity -> 1.
    ppl = quality.perplexity(sum_logprob=-0.001, n_tokens=100)
    assert ppl == pytest.approx(1.0, abs=1e-2)


def test_perplexity_matches_definition():
    sum_logprob, n_tokens = -50.0, 25
    assert quality.perplexity(sum_logprob, n_tokens) == pytest.approx(
        math.exp(-sum_logprob / n_tokens)
    )


def test_perplexity_rejects_zero_tokens():
    with pytest.raises(ValueError, match="n_tokens"):
        quality.perplexity(sum_logprob=-1.0, n_tokens=0)


def test_passes_uses_explicit_threshold_over_module_default():
    assert quality.passes(0.9, threshold=0.85) is True
    assert quality.passes(0.8, threshold=0.85) is False


def test_passes_raises_if_no_threshold_set_anywhere():
    # Deliberate: §4.4 requires the threshold decided *before* the ladder
    # runs. Falling through to some default here would let an unset
    # threshold pass silently, which is the exact failure mode §4.4 warns
    # against ("a threshold chosen after seeing the numbers is not a
    # threshold").
    with pytest.raises(ValueError, match="threshold not set"):
        quality.passes(0.99)


def test_passes_uses_module_level_threshold_once_set(monkeypatch):
    monkeypatch.setattr(quality, "EXACT_MATCH_THRESHOLD", 0.95)
    assert quality.passes(0.97) is True
    assert quality.passes(0.90) is False


def _write_jsonl(path, records):
    path.write_text("\n".join(json.dumps(r) for r in records))


def test_check_precision_end_to_end(tmp_path):
    ref = tmp_path / "fp16.jsonl"
    cand = tmp_path / "int4.jsonl"
    _write_jsonl(
        ref,
        [{"prompt_id": "p1", "output": "hello"}, {"prompt_id": "p2", "output": "world"}],
    )
    _write_jsonl(
        cand,
        [{"prompt_id": "p1", "output": "hello"}, {"prompt_id": "p2", "output": "wrld"}],
    )

    result = quality.check_precision(cand, ref, threshold=0.6)

    assert result == {
        "checked": True,
        "reference": "fp16",
        "exact_match": 0.5,
        "passed": False,  # 0.5 < 0.6 threshold
    }


def test_check_precision_rejects_mismatched_prompt_ids(tmp_path):
    ref = tmp_path / "fp16.jsonl"
    cand = tmp_path / "int4.jsonl"
    _write_jsonl(ref, [{"prompt_id": "p1", "output": "hello"}])
    _write_jsonl(cand, [{"prompt_id": "p2", "output": "hello"}])

    with pytest.raises(ValueError, match="different prompt_ids"):
        quality.check_precision(cand, ref, threshold=0.5)
