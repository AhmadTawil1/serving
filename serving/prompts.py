"""Fixed prompt set for the concurrency/precision sweep.

Frozen today (Day 18) and never edited again per M03-SERVING.md §9 -- editing
the prompt set after any measurement invalidates every prior comparison,
since prefill cost is prompt-length-dependent (§3, "held fixed").

Design: a small set of realistic, topically-varied instruct prompts (systems
explanation, code, reasoning, incident writing) each padded with a fixed,
deterministic filler passage to land at exactly `config.INPUT_TOKENS` tokens
under the study model's own tokenizer. Content matters for the Day 22
quality gate (exact-match / perplexity against an FP16 reference needs real
text, not noise); the padding exists only to control prefill cost, and is
never itself the thing being asked about.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from . import config

PROMPTS_PATH = Path("data/prompts.json")

_SEEDS = [
    "Explain how PagedAttention avoids KV-cache fragmentation compared to a "
    "naive contiguous allocation, in terms a new ML infra engineer would "
    "understand.",
    "Write a Python function that merges two sorted linked lists in place, "
    "and explain its time complexity.",
    "Summarize the tradeoffs between closed-loop and open-loop load testing "
    "for a concurrent request-serving system.",
    "A shop sells apples in bags of 4 and oranges in bags of 6. A customer "
    "wants exactly 24 apples and 24 oranges using the fewest bags possible. "
    "Work through the reasoning step by step.",
    "Draft a short incident postmortem for a service that ran out of GPU "
    "memory under load, covering root cause, impact, and two concrete "
    "follow-up actions.",
    "Explain the difference between TTFT and TPOT when benchmarking an LLM "
    "inference server, and why they are dominated by different parts of the "
    "request lifecycle.",
    "Describe, at a high level, what changes inside a transformer decoder "
    "when it switches from FP16 to INT4 weight-only quantization.",
    "Write a short product description for a fictional noise-cancelling "
    "mechanical keyboard aimed at open-plan offices.",
]

# Deterministic filler, tiled and truncated to pad every seed out to
# INPUT_TOKENS. Fixed text, not random, so the padded prompt is byte-identical
# on every machine that runs this file.
_FILLER = (
    "For background, distributed-systems research has long studied the "
    "tradeoff between throughput and tail latency under load, and serving "
    "frameworks make different choices about batching, scheduling, and "
    "memory management that shift where that tradeoff sits. This sentence "
    "is repeated only to occupy context length and carries no additional "
    "information beyond this point. "
)


def _pad_to_length(text: str, tokenizer, n_tokens: int) -> str:
    """Pad or truncate `text` so it tokenizes to exactly `n_tokens` under
    `tokenizer`.

    Padding by tiling the filler onto the *string*, not by concatenating
    token-id lists: BPE tokenizers can merge across an id-list join point
    (e.g. a sentence-ending "." plus a following word can retokenize as one
    token once decoded back to text), so id concatenation silently loses
    tokens at every join and the result stops being exactly `n_tokens` long.
    Tiling the string sidesteps the join issue but decode(ids[:n_tokens])
    can still re-encode to a slightly different count at the single cut
    point, so the last step converges on the exact count explicitly rather
    than assuming the round-trip is stable.
    """
    if len(tokenizer(text, add_special_tokens=False)["input_ids"]) > n_tokens:
        ids = tokenizer(text, add_special_tokens=False)["input_ids"][:n_tokens]
        return tokenizer.decode(ids)

    full_text = text
    while len(tokenizer(full_text, add_special_tokens=False)["input_ids"]) < n_tokens + 16:
        full_text += " " + _FILLER
    ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]

    target = n_tokens
    for _ in range(10):
        candidate = tokenizer.decode(ids[:target])
        actual = len(tokenizer(candidate, add_special_tokens=False)["input_ids"])
        if actual == n_tokens:
            return candidate
        target += n_tokens - actual
    raise RuntimeError(
        f"could not converge prompt to exactly {n_tokens} tokens (last actual={actual})"
    )


def build_prompts(tokenizer, n_tokens: int = config.INPUT_TOKENS) -> list[str]:
    """Regenerate the padded prompt set from `_SEEDS`. Deterministic given a
    tokenizer and `n_tokens`; used only to (re)write `PROMPTS_PATH` via
    `freeze()` -- measurement code reads the frozen file, never calls this.
    """
    return [_pad_to_length(seed, tokenizer, n_tokens) for seed in _SEEDS]


def freeze(tokenizer) -> dict:
    """Write the frozen prompt set to disk. Run once, on Day 18."""
    padded = build_prompts(tokenizer)
    token_counts = [
        len(tokenizer(p, add_special_tokens=False)["input_ids"]) for p in padded
    ]
    payload = {
        "tokenizer": config.STUDY_MODEL,
        "input_tokens": config.INPUT_TOKENS,
        "token_counts": token_counts,
        "prompts": padded,
    }
    PROMPTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    PROMPTS_PATH.write_text(json.dumps(payload, indent=2))
    return payload


def load_prompts() -> list[str]:
    data = json.loads(PROMPTS_PATH.read_text())
    return data["prompts"]


def prompts_sha() -> str:
    """SHA1 over the frozen prompts file's exact bytes -- `prompts_sha` in
    every record's `prov` block (§8). Hashed here, and only here -- imported
    by callers rather than recomputed, per the M01 lesson in REFERENCE.md §7.
    """
    return hashlib.sha1(PROMPTS_PATH.read_bytes()).hexdigest()
