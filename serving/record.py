"""Per-repeat result records, matching M03-SERVING.md §8's schema exactly.

One JSON line per repeat -- never pre-aggregated at write time, so
run-to-run spread stays recoverable and a crossover inside the noise band
can be shown, not just asserted (REFERENCE.md §7, carried from M01/M02:
"One record per repeat, aggregated at read time").

`build_record` is the single place that turns a window's raw
`client.RequestResult` list into the schema's derived fields (throughput,
p50/p95, goodput) -- REFERENCE.md §7's "a provenance check must call the
same function that wrote the field" applies here just as much as to a hash.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from . import config


def percentile(values: list[float], p: float) -> float:
    if not values:
        raise ValueError("no values to compute a percentile over")
    return float(np.percentile(values, p))


def config_id(cfg: dict) -> str:
    """SHA1 over the sorted config -- identical configs always hash the same
    regardless of key insertion order (same pattern as Retrieval's grid
    `config_id`, M01 LOG.md Day 2).
    """
    blob = json.dumps(cfg, sort_keys=True).encode()
    return "sha1:" + hashlib.sha1(blob).hexdigest()


def _goodput_req_s(request_results: list, window_s: float) -> float:
    """Requests/sec meeting the SLO (TTFT and TPOT both under threshold) --
    the number an operator actually buys, per §8's metric table.
    """
    if window_s <= 0:
        return 0.0
    n_good = sum(
        1
        for r in request_results
        if r.ttft_s * 1000 < config.SLO_TTFT_MS and r.tpot_s * 1000 < config.SLO_TPOT_MS
    )
    return n_good / window_s


def build_record(
    *,
    cfg: dict,
    request_results: list,
    repeat: int,
    of_repeats: int,
    warmup_requests: int,
    window_s: float,
    prov: dict,
    cost: dict | None = None,
    build: dict | None = None,
    quality: dict | None = None,
    resources: dict | None = None,
    seed: int = 1337,
    status: str = "ok",
) -> dict[str, Any]:
    """Assemble one record from one `client.run_measurement_window` call's
    results. Every derived statistic (throughput, p50/p95, goodput) is
    computed here and only here.
    """
    if not request_results:
        raise ValueError("no request results in this window -- nothing to record")

    ttft_ms = [r.ttft_s * 1000 for r in request_results]
    tpot_ms = [r.tpot_s * 1000 for r in request_results]
    e2e_ms = [r.e2e_s * 1000 for r in request_results]
    total_tokens = sum(r.n_tokens for r in request_results)
    output_tok_s = total_tokens / window_s if window_s > 0 else 0.0

    return {
        "config": cfg,
        "config_id": config_id(cfg),
        "throughput": {
            "output_tok_s": output_tok_s,
            "goodput_req_s": _goodput_req_s(request_results, window_s),
            "slo": {"ttft_ms": config.SLO_TTFT_MS, "tpot_ms": config.SLO_TPOT_MS},
        },
        "latency": {
            "ttft_ms": {"p50": percentile(ttft_ms, 50), "p95": percentile(ttft_ms, 95)},
            "tpot_ms": {"p50": percentile(tpot_ms, 50), "p95": percentile(tpot_ms, 95)},
            "e2e_ms": {"p50": percentile(e2e_ms, 50), "p95": percentile(e2e_ms, 95)},
        },
        "cost": cost
        or {
            "usd_per_hour": None,
            "price_source": None,
            "price_date": None,
            "tok_per_usd_hour": None,
        },
        "resources": resources
        or {
            "peak_vram_mb": None,
            "kv_cache_util_pct": None,
            "gpu_util_pct": None,
            "client_cpu_pct": None,
        },
        "build": build or {"engine_build_s": 0, "engine_id": None},
        "quality": quality
        or {"checked": False, "reference": None, "exact_match": None, "passed": None},
        "run": {
            "n_requests": len(request_results),
            "warmup_requests": warmup_requests,
            "window_s": window_s,
            "repeat": repeat,
            "of_repeats": of_repeats,
            "seed": seed,
            "status": status,
        },
        "prov": prov,
    }


def write_record(path: Path, record: dict) -> None:
    """Append one JSON line. Never overwrites -- a sweep that crashes and
    resumes should not lose the records it already wrote.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(record) + "\n")


def read_records(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
