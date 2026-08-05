"""Parity check (§11 / "Day 20"): before any concurrency sweep runs against
a freshly built TensorRT-LLM engine, confirm it is serving the *same model*
vLLM is -- same prompts, greedy, compared token-for-token. "Divergence
beyond sampling noise means the engine is not serving the same model --
stop and fix, because every number after this depends on it."

Reuses `serving/quality.py`'s exact-match machinery: this is the same
question Day 22's precision-ladder quality gate asks (does framework/config
X produce the same output as a trusted reference), just asked once, early,
between frameworks rather than across precisions -- REFERENCE.md §7's "do
not reimplement" rule applies to comparison logic, not only hashes.

Works against any two OpenAI-compatible endpoints, so it runs unmodified
against two `mock_server.py` instances for testing (`tests/test_check_parity
.py`) and against a live vLLM/TensorRT-LLM pair on the GPU host for the
real Day 20 gate:

    python scripts/check_parity.py --port-a 8000 --port-b 8001 \\
        --label-a vllm --label-b tensorrt-llm
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from serving import config, prompts, quality  # noqa: E402

_LABEL = "check_parity"


def fetch_completion_text(port: int, prompt: str) -> str:
    """Greedy completion text from one server, via the same streamed
    OpenAI-compatible endpoint everything else in this harness uses --
    accumulates the streamed text pieces rather than requesting a
    non-streaming response, so this needs no server-side code path the rest
    of the study doesn't already exercise.
    """
    url = f"http://localhost:{port}/v1/completions"
    payload = {
        "model": config.STUDY_MODEL,
        "prompt": prompt,
        "max_tokens": config.OUTPUT_TOKENS,
        "min_tokens": config.OUTPUT_TOKENS,
        "temperature": 0,
        "stream": True,
    }
    pieces: list[str] = []
    with requests.post(url, json=payload, stream=True, timeout=120) as resp:
        resp.raise_for_status()
        for line in resp.iter_lines():
            if not line or not line.startswith(b"data: "):
                continue
            data = line[len(b"data: ") :]
            if data == b"[DONE]":
                break
            chunk = json.loads(data)
            pieces.append(chunk["choices"][0].get("text", ""))
    return "".join(pieces)


def check(port_a: int, port_b: int, label_a: str, label_b: str) -> dict:
    prompt_set = prompts.load_prompts()
    outputs_a = [fetch_completion_text(port_a, p) for p in prompt_set]
    outputs_b = [fetch_completion_text(port_b, p) for p in prompt_set]

    em = quality.exact_match_rate(outputs_a, outputs_b)
    # Unlike the Day 22 precision-ladder gate (which allows some
    # quantization-induced drift below a chosen threshold, §4.4), this is
    # two servers claiming to run the *identical* FP16 model -- anything
    # short of exact match means one of them isn't, so there is no
    # threshold to tune here.
    passed = em == 1.0

    print(f"[{_LABEL}] {label_a} vs {label_b}: exact_match={em:.3f} over {len(prompt_set)} prompts")
    if not passed:
        for i, (a, b) in enumerate(zip(outputs_a, outputs_b)):
            if a.strip() != b.strip():
                print(f"[{_LABEL}]   prompt {i} diverges:")
                print(f"[{_LABEL}]     {label_a}: {a[:200]!r}")
                print(f"[{_LABEL}]     {label_b}: {b[:200]!r}")
    print(f"[{_LABEL}] {'PASS' if passed else 'FAIL -- stop and fix before any sweep'}")

    return {"exact_match": em, "passed": passed}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port-a", type=int, required=True)
    ap.add_argument("--port-b", type=int, required=True)
    ap.add_argument("--label-a", default="vllm")
    ap.add_argument("--label-b", default="tensorrt-llm")
    args = ap.parse_args()

    result = check(args.port_a, args.port_b, args.label_a, args.label_b)
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
