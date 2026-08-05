"""§4.2's validity gate: read a sweep run against `mock_server.py` and check
that the client held concurrency up to the top of the axis without becoming
the bottleneck itself.

Two checks, not one:

1. **Ratio to the mock server's theoretical ceiling** -- informational.
   On this Windows dev box it sits around 0.6 at every level rather than
   ~1.0, because Windows' default asyncio timer granularity inflates each
   short `asyncio.sleep()` beyond its nominal duration (measured: a
   configured 10ms inter-token sleep lands at ~16-17ms). That is a property
   of the mock *server's* own timing fidelity on this OS, not the client --
   see LOG.md Day 2 for the measurement that pins this down.
2. **Ratio to linear scaling from the concurrency-1 baseline** -- the real
   gate. If the client were the bottleneck at high concurrency, this ratio
   would degrade as concurrency increases (a plateau). A ratio that stays
   flat within noise all the way to 128 is the actual "not the bottleneck"
   signature, and is independent of the server's absolute timing accuracy.

    python scripts/validate_client.py --records data/mock_validity_gate.jsonl
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import mock_server  # noqa: E402
from serving import config, record  # noqa: E402

LINEAR_RATIO_PASS_BAND = (0.85, 1.15)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", required=True)
    args = ap.parse_args()

    records = record.read_records(Path(args.records))
    by_level: dict[int, list[dict]] = {}
    for r in records:
        by_level.setdefault(r["config"]["concurrency"], []).append(r)

    levels = sorted(by_level)
    baseline = sum(r["throughput"]["output_tok_s"] for r in by_level[levels[0]]) / len(
        by_level[levels[0]]
    )

    header = f"{'concurrency':>11} {'achieved':>10} {'theoretical':>12} {'theo_ratio':>10} {'linear_ratio':>12} {'cpu_mean':>9}"
    print(header)

    all_pass = True
    for c in levels:
        recs = by_level[c]
        achieved = sum(r["throughput"]["output_tok_s"] for r in recs) / len(recs)
        theoretical = mock_server.theoretical_ceiling_tok_s(c, output_tokens=config.OUTPUT_TOKENS)
        theo_ratio = achieved / theoretical
        linear_ratio = achieved / (baseline * c)
        cpu_mean = sum(r["resources"]["client_cpu_pct"] for r in recs) / len(recs)

        lo, hi = LINEAR_RATIO_PASS_BAND
        ok = lo <= linear_ratio <= hi
        all_pass = all_pass and ok
        flag = "" if ok else "  <-- OUT OF BAND"
        print(
            f"{c:>11} {achieved:>10.1f} {theoretical:>12.1f} {theo_ratio:>10.2f} "
            f"{linear_ratio:>12.3f} {cpu_mean:>9.1f}{flag}"
        )

    print()
    top = levels[-1]
    verdict = "PASS" if all_pass else "FAIL"
    print(
        f"{verdict}: linear-scaling ratio stayed within "
        f"{LINEAR_RATIO_PASS_BAND} at every level through concurrency {top}."
    )


if __name__ == "__main__":
    main()
