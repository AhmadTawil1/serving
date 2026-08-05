"""Per-precision quality gate (M03-SERVING.md §4.4).

Before a precision's throughput is allowed into the headline figure, its
greedy output on the fixed prompt set is compared against the FP16 output
from the **same card and framework** (not cross-card, not cross-framework --
the gate isolates precision as the only variable). A precision that fails is
reported as *rejected on quality*, with the number, and excluded from the
throughput figure rather than quietly omitted.

The threshold is decided on Day 22, before the ladder runs, per §4.4 -- "a
threshold chosen after seeing the numbers is not a threshold." It is
deliberately left unset here (`None`) rather than given a plausible-looking
default, so a stale default can't silently stand in for a real decision.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

EXACT_MATCH_THRESHOLD: float | None = None  # set on Day 22, in LOG.md and here


def exact_match_rate(candidates: list[str], references: list[str]) -> float:
    """Fraction of (candidate, reference) pairs that match exactly after
    stripping surrounding whitespace. Valid because decoding is greedy
    (temperature=0, §3) -- two runs of the same precision should be
    byte-identical, so any candidate/reference divergence is attributable to
    the precision change, not sampling noise.
    """
    if not candidates:
        raise ValueError("no outputs to compare")
    if len(candidates) != len(references):
        raise ValueError(
            f"candidate/reference count mismatch: {len(candidates)} vs {len(references)}"
        )
    matches = sum(c.strip() == r.strip() for c, r in zip(candidates, references))
    return matches / len(candidates)


def perplexity(sum_logprob: float, n_tokens: int) -> float:
    """Perplexity from a summed log-probability over `n_tokens`, as returned
    by the served OpenAI-compatible endpoint's per-token logprobs. Fallback
    metric for when exact-match is too strict to be informative (e.g. a
    precision that is fluent but takes a different greedy path early and
    diverges token-by-token thereafter).
    """
    if n_tokens <= 0:
        raise ValueError("n_tokens must be positive")
    return math.exp(-sum_logprob / n_tokens)


def passes(exact_match: float, threshold: float | None = None) -> bool:
    threshold = EXACT_MATCH_THRESHOLD if threshold is None else threshold
    if threshold is None:
        raise ValueError(
            "quality threshold not set -- decide it on Day 22, before the "
            "precision ladder runs (§4.4), and set quality.EXACT_MATCH_THRESHOLD"
        )
    return exact_match >= threshold


def _load_outputs(path: Path) -> dict[str, str]:
    """path: JSONL, one `{'prompt_id': ..., 'output': str}` per line."""
    out: dict[str, str] = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        out[rec["prompt_id"]] = rec["output"]
    return out


def check_precision(
    candidate_path: Path, reference_path: Path, threshold: float | None = None
) -> dict:
    """Compare a precision's greedy outputs against the FP16 reference on the
    same card and framework. Returns the `quality` block for the record
    schema in §8.
    """
    candidate = _load_outputs(candidate_path)
    reference = _load_outputs(reference_path)
    if candidate.keys() != reference.keys():
        raise ValueError("candidate and reference cover different prompt_ids")

    ids = sorted(reference)
    em = exact_match_rate([candidate[i] for i in ids], [reference[i] for i in ids])
    return {
        "checked": True,
        "reference": "fp16",
        "exact_match": em,
        "passed": passes(em, threshold),
    }
