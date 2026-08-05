"""Day 4 ("Day 21") hard stop, made mechanical rather than a judgment call
made live on a disconnecting Colab session.

Reads `scripts/build_trtllm_engine.py`'s build-log JSONL, takes the latest
FP16 build status per card, and prints the exact `SCOPE.md` → "The Day 21
engine hard stop" verdict for that state -- the four-way rule (both
serving / A100 only / L4 only / neither) is written down there and here
identically on purpose, so this script can't drift from the policy it is
applying.

    python scripts/check_hard_stop.py --log results/trtllm_build_log.jsonl
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from serving import config, record  # noqa: E402

_LABEL = "check_hard_stop"

_ACTIONS = {
    "both_serving": (
        "Proceed with the full TensorRT-LLM axis on both cards -- both FP16 "
        "concurrency curves this session if time remains (M03-SERVING.md §11)."
    ),
    "a100_only": (
        "SCOPE.md fallback ('A100 serving, L4 not'): report the A100 crossover; "
        "report the L4 build failure with its exact error as a portability "
        "result, not a gap; add a vLLM-only L4 concurrency curve so the card "
        "axis still has data."
    ),
    "l4_only": (
        "SCOPE.md fallback ('L4 serving, A100 not'), applied symmetrically to "
        "the §17-named case: report the L4 crossover; report the A100 build "
        "failure with its exact error; add a vLLM-only A100 concurrency curve."
    ),
    "neither": (
        "§6 hard stop: the TensorRT-LLM axis is dropped ENTIRELY. Execute the "
        "Day 18 contingency (M03-SERVING.md §17, 'TensorRT-LLM will not install "
        "on the runtime'): switch the framework axis to vLLM configuration "
        "knobs (KV-cache fraction, max-num-seqs, chunked prefill on/off) -- "
        "still a concurrency sweep, still two cards, still a crossover "
        "question. Rewrite SCOPE.md now, not later."
    ),
}


def latest_status_per_card(records: list[dict], precision: str = "fp16") -> dict[str, bool]:
    """{card: succeeded} from the build log, keeping only the *latest*
    record per card at the given precision (a card may have been attempted
    more than once; the most recent attempt is the one that matters for the
    hard-stop decision, per §6's build log being append-only chronological).
    """
    status: dict[str, bool] = {}
    for r in records:
        if r["precision"] != precision:
            continue
        status[r["card"]] = r["succeeded"]
    return status


def decide(a100_succeeded: bool, l4_succeeded: bool) -> dict:
    if a100_succeeded and l4_succeeded:
        verdict = "both_serving"
    elif a100_succeeded and not l4_succeeded:
        verdict = "a100_only"
    elif l4_succeeded and not a100_succeeded:
        verdict = "l4_only"
    else:
        verdict = "neither"
    return {"verdict": verdict, "action": _ACTIONS[verdict]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default="results/trtllm_build_log.jsonl")
    ap.add_argument("--a100-card", default=config.CARDS[0])
    ap.add_argument("--l4-card", default=config.CARDS[1])
    ap.add_argument("--precision", default="fp16")
    args = ap.parse_args()

    log_path = Path(args.log)
    records = record.read_records(log_path)
    if not records:
        print(f"[{_LABEL}] no build records found at {log_path} -- nothing has been attempted yet")
        raise SystemExit(1)

    status = latest_status_per_card(records, args.precision)
    a100_ok = status.get(args.a100_card, False)
    l4_ok = status.get(args.l4_card, False)

    result = decide(a100_ok, l4_ok)
    print(f"[{_LABEL}] {args.a100_card} engine serving: {a100_ok}")
    print(f"[{_LABEL}] {args.l4_card} engine serving: {l4_ok}")
    print(f"[{_LABEL}] verdict: {result['verdict']}")
    print(f"[{_LABEL}] action: {result['action']}")


if __name__ == "__main__":
    main()
