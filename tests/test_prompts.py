"""Unit tests for prompts.py's padding/truncation and freeze/hash pipeline.

Uses a fake, offline, whitespace-level tokenizer instead of downloading the
study model's real tokenizer -- keeps this suite fast and network-free, same
discipline as test_pins.py. Two of the tests below exist specifically
because of a real bug caught while writing this file: `tokenizer.decode(ids
[:n])` is not guaranteed to re-encode to exactly `n` tokens (a BPE merge can
form across the cut point), which silently produced 504-token prompts
against a 512 target on the first pass. `_FlakyTokenizer` reproduces that
class of bug deterministically so the fix (an explicit convergence retry) is
covered by something other than "it happened to work on Qwen's tokenizer."
"""

from __future__ import annotations

import hashlib
import json

import pytest

from serving import prompts


class _WordTokenizer:
    """Deterministic whitespace tokenizer. Stable round-trip (no merge
    artifacts) -- enough to exercise the padding/truncation control flow
    without a network call.
    """

    def __init__(self):
        self._id_to_word: dict[int, str] = {}
        self._word_to_id: dict[str, int] = {}

    def _id_for(self, word: str) -> int:
        if word not in self._word_to_id:
            i = len(self._word_to_id)
            self._word_to_id[word] = i
            self._id_to_word[i] = word
        return self._word_to_id[word]

    def __call__(self, text: str, add_special_tokens: bool = False) -> dict:
        return {"input_ids": [self._id_for(w) for w in text.split()]}

    def decode(self, ids: list[int]) -> str:
        return " ".join(self._id_to_word[i] for i in ids)


class _FlakyTokenizer(_WordTokenizer):
    """Merges the last two words into one (no separating space) whenever a
    slice of exactly `flaky_at` tokens is decoded -- simulating a BPE merge
    across a truncation boundary. `once=True` fires the merge a single time
    (models a one-off boundary artifact, which the retry loop should recover
    from); `once=False` fires every time that exact length is decoded again
    (models a length that can never land cleanly, which should exhaust the
    retry budget).
    """

    def __init__(self, flaky_at: int, once: bool):
        super().__init__()
        self.flaky_at = flaky_at
        self.once = once
        self._triggered = False

    def decode(self, ids: list[int]) -> str:
        words = [self._id_to_word[i] for i in ids]
        should_merge = len(ids) == self.flaky_at and len(words) >= 2
        if should_merge and not (self.once and self._triggered):
            self._triggered = True
            words = words[:-2] + [words[-2] + words[-1]]
        return " ".join(words)


@pytest.fixture(autouse=True)
def isolated_prompts_path(tmp_path, monkeypatch):
    monkeypatch.setattr(prompts, "PROMPTS_PATH", tmp_path / "prompts.json")


def test_pad_to_length_truncates_long_seed():
    tok = _WordTokenizer()
    text = " ".join(f"w{i}" for i in range(100))

    result = prompts._pad_to_length(text, tok, n_tokens=10)

    assert len(tok(result, add_special_tokens=False)["input_ids"]) == 10
    assert result == " ".join(f"w{i}" for i in range(10))


def test_pad_to_length_pads_short_seed_to_exact_count():
    tok = _WordTokenizer()

    result = prompts._pad_to_length("hello world", tok, n_tokens=25)

    assert len(tok(result, add_special_tokens=False)["input_ids"]) == 25


def test_pad_to_length_converges_despite_one_time_merge_artifact():
    n_tokens = 30
    tok = _FlakyTokenizer(flaky_at=n_tokens, once=True)

    result = prompts._pad_to_length("seed text", tok, n_tokens=n_tokens)

    assert len(tok(result, add_special_tokens=False)["input_ids"]) == n_tokens
    assert tok._triggered  # the merge path was actually exercised, not skipped


def test_pad_to_length_raises_if_it_never_converges():
    n_tokens = 30
    tok = _FlakyTokenizer(flaky_at=n_tokens, once=False)

    with pytest.raises(RuntimeError, match="could not converge"):
        prompts._pad_to_length("seed text", tok, n_tokens=n_tokens)


def test_freeze_writes_exact_token_counts_and_hashable_file():
    tok = _WordTokenizer()

    payload = prompts.freeze(tok)

    assert len(payload["prompts"]) == len(prompts._SEEDS)
    assert payload["token_counts"] == [prompts.config.INPUT_TOKENS] * len(prompts._SEEDS)
    for p in payload["prompts"]:
        assert len(tok(p, add_special_tokens=False)["input_ids"]) == prompts.config.INPUT_TOKENS


def test_load_prompts_roundtrips_freeze_output():
    tok = _WordTokenizer()
    payload = prompts.freeze(tok)

    loaded = prompts.load_prompts()

    assert loaded == payload["prompts"]


def test_prompts_sha_matches_independent_hash_of_the_frozen_file():
    tok = _WordTokenizer()
    prompts.freeze(tok)

    expected = hashlib.sha1(prompts.PROMPTS_PATH.read_bytes()).hexdigest()
    assert prompts.prompts_sha() == expected


def test_prompts_sha_changes_if_the_frozen_file_changes():
    tok = _WordTokenizer()
    prompts.freeze(tok)
    first = prompts.prompts_sha()

    data = json.loads(prompts.PROMPTS_PATH.read_text())
    data["prompts"][0] += " extra"
    prompts.PROMPTS_PATH.write_text(json.dumps(data))

    assert prompts.prompts_sha() != first
