"""Concurrency-sweep driver (§4.3, §8): warmup at the target concurrency
(discarded), then a fixed measurement window, repeated `--repeats` times,
one record written per repeat via `serving/record.py`.

Framework-agnostic -- it only ever talks to `--base-url` over the
OpenAI-compatible completions endpoint, so the exact same invocation drives
the mock server (`mock_server.py`, the §4.2 validity gate, real today on the
CPU dev box), vLLM, and TensorRT-LLM (both GPU-host-only, later).

    python scripts/sweep.py --base-url http://localhost:8000 \\
        --framework vllm --card A100-SXM4-40GB --precision fp16 \\
        --out data/vllm_a100_fp16.jsonl
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import client as client_mod  # noqa: E402  (scripts/ is on sys.path[0] when run directly)
from serving import config, prompts as prompts_mod, provenance, record  # noqa: E402

# "mock" is not a real framework/card -- it is the §4.2 validity-gate target,
# allowed here so the exact same driver exercises it, not a parallel script.
_FRAMEWORK_CHOICES = [*config.FRAMEWORKS, "mock"]
_CARD_CHOICES = [*config.CARDS, "mock"]


def _warmup_count_per_worker(warmup_requests_total: int, concurrency: int) -> int:
    """§4.3 and §18 specify a *total* warmup request count (working
    assumption: 32), not a per-worker count -- distributed round-robin
    across the `concurrency` workers so every worker's connection is
    exercised at least once. At concurrency 128 this means each worker gets
    very few dedicated warmup requests; whether that is enough to clear
    first-request engine artifacts is exactly what §18 asks to confirm
    against a real curve, not something assumed correct here.
    """
    return max(1, warmup_requests_total // concurrency)


async def run_one_level(
    *,
    base_url: str,
    concurrency: int,
    prompt_set: list[str],
    framework: str,
    framework_version: str | None,
    card: str,
    precision: str,
    warmup_requests_total: int,
    window_s: float,
    repeats: int,
    out_path: Path,
    prov: dict,
) -> None:
    n_per_worker = _warmup_count_per_worker(warmup_requests_total, concurrency)
    print(
        f"[sweep] concurrency={concurrency}: warmup "
        f"({n_per_worker} req/worker x {concurrency} workers)"
    )
    await client_mod.run_warmup(base_url, concurrency, prompt_set, n_per_worker)

    cfg = {
        "framework": framework,
        "framework_version": framework_version,
        "card": card,
        "concurrency": concurrency,
        "precision": precision,
        "model": config.STUDY_MODEL,
        "max_model_len": config.MAX_MODEL_LEN,
        "input_tokens": config.INPUT_TOKENS,
        "output_tokens": config.OUTPUT_TOKENS,
        "greedy": True,
    }

    for repeat in range(1, repeats + 1):
        print(
            f"[sweep] concurrency={concurrency}: repeat {repeat}/{repeats}, "
            f"window={window_s}s"
        )
        window = await client_mod.run_measurement_window(
            base_url, concurrency, prompt_set, window_s
        )
        resources = {
            "peak_vram_mb": None,
            "kv_cache_util_pct": None,
            "gpu_util_pct": None,
            "client_cpu_pct": window.client_cpu_pct["mean"],
        }
        rec = record.build_record(
            cfg=cfg,
            request_results=window.request_results,
            repeat=repeat,
            of_repeats=repeats,
            warmup_requests=n_per_worker * concurrency,
            window_s=window.wall_s,
            prov=prov,
            resources=resources,
        )
        record.write_record(out_path, rec)
        print(
            f"[sweep]   n_requests={window.n_requests} "
            f"output_tok_s={rec['throughput']['output_tok_s']:.1f} "
            f"ttft_p50={rec['latency']['ttft_ms']['p50']:.1f}ms "
            f"client_cpu(mean/max)={window.client_cpu_pct['mean']:.1f}/"
            f"{window.client_cpu_pct['max']:.1f}%"
        )


async def run_sweep(args: argparse.Namespace) -> None:
    prompt_set = prompts_mod.load_prompts()
    prov = provenance.stamp(config.STUDY_MODEL)
    out_path = Path(args.out)
    for concurrency in args.levels:
        await run_one_level(
            base_url=args.base_url,
            concurrency=concurrency,
            prompt_set=prompt_set,
            framework=args.framework,
            framework_version=args.framework_version,
            card=args.card,
            precision=args.precision,
            warmup_requests_total=args.warmup_requests,
            window_s=args.window_s,
            repeats=args.repeats,
            out_path=out_path,
            prov=prov,
        )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--framework", required=True, choices=_FRAMEWORK_CHOICES)
    ap.add_argument("--framework-version", default=None)
    ap.add_argument("--card", required=True, choices=_CARD_CHOICES)
    ap.add_argument("--precision", default="fp16", choices=config.PRECISIONS)
    ap.add_argument(
        "--levels", type=int, nargs="+", default=config.CONCURRENCY_LEVELS
    )
    ap.add_argument("--window-s", type=float, default=60.0)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--warmup-requests", type=int, default=32)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    asyncio.run(run_sweep(args))


if __name__ == "__main__":
    main()
