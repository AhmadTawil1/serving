"""Compute every number Paper 3 reports, from the result files.

**The point of this script is that no figure in the paper is typed by hand.**
The Day 8 verification pass (`LOG.md`) found three numbers in the working notes
that were wrong -- a run-to-run spread quoted for one curve and applied to
four, a throughput ratio quoted as a single value when it varied 4.35-5.26x
across operating points, and a "triples" that was 2.95x. None changed a
conclusion, but all three were prose drifting away from the data it described.

So: `analysis.py --emit markdown` prints the tables, and those tables go into
`PAPER.md` verbatim. If a number appears in the paper and not in this script's
output, it does not belong there.

Usage:

    python scripts/analysis.py                       # human-readable report
    python scripts/analysis.py --emit markdown       # paste-ready tables
    python scripts/analysis.py --price-a100 3.67 --price-l4 0.70
    python scripts/analysis.py --sensitivity 0.30    # +/-30% price sweep

Prices are **arguments, not constants**, because they are not measured. Their
provenance (vendor, date, source) is the analyst's to record in Method; this
script only reports what follows from them, and how far they can move before
the conclusion changes.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# (label, filename, card, prefix_caching) -- the four curves of the study.
CURVES = [
    ("A100 off", "vllm_a100_fp16_nocache.jsonl", "A100", False),
    ("A100 on", "vllm_a100_fp16_prefixcache_on.jsonl", "A100", True),
    ("L4 off", "vllm_l4_fp16_nocache.jsonl", "L4", False),
    ("L4 on", "vllm_l4_fp16_cache.jsonl", "L4", True),
]

# Frozen inputs that MUST be identical across all four curves, or the curves
# are not comparable and nothing below means anything. Asserted, not assumed --
# this is the check that would have caught the Day 8 stale-server file if the
# record had carried the server's own configuration rather than the launcher's.
CONFIG_INVARIANTS = ["model", "max_model_len", "input_tokens", "output_tokens", "greedy", "precision"]
PROV_INVARIANTS = ["model_revision", "prompts_sha"]


def load(results_dir: Path) -> dict[str, dict[int, list[dict]]]:
    """Return {curve_label: {concurrency: [record, ...]}}."""
    out: dict[str, dict[int, list[dict]]] = {}
    for label, fname, _card, _pc in CURVES:
        path = results_dir / fname
        if not path.exists():
            raise SystemExit(f"missing result file: {path}")
        by_conc: dict[int, list[dict]] = defaultdict(list)
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            by_conc[rec["config"]["concurrency"]].append(rec)
        out[label] = dict(by_conc)
    return out


def check_integrity(data: dict[str, dict[int, list[dict]]]) -> list[str]:
    """Return a list of human-readable findings. Empty means everything holds.

    Deliberately returns findings rather than raising: a mismatch is something
    to report in Threats, not necessarily something to crash on.
    """
    findings: list[str] = []

    for label, by_conc in data.items():
        n = sum(len(v) for v in by_conc.values())
        statuses = {r["run"]["status"] for v in by_conc.values() for r in v}
        if statuses != {"ok"}:
            findings.append(f"{label}: non-ok statuses present: {statuses}")
        reps = {len(v) for v in by_conc.values()}
        if len(reps) != 1:
            findings.append(f"{label}: uneven repeat counts across levels: {sorted(reps)}")
        findings.append(f"[info] {label}: {n} records, {len(by_conc)} levels, {reps.pop()} repeats")

    # Every curve must sweep the same concurrency levels.
    level_sets = {label: tuple(sorted(by_conc)) for label, by_conc in data.items()}
    if len({v for v in level_sets.values()}) != 1:
        findings.append(f"concurrency levels differ across curves: {level_sets}")

    # The frozen inputs.
    first = next(iter(data.values()))
    ref = first[min(first)][0]
    for field in CONFIG_INVARIANTS:
        vals = {label: by_conc[min(by_conc)][0]["config"][field] for label, by_conc in data.items()}
        if len(set(vals.values())) != 1:
            findings.append(f"**CONTROL BROKEN** config.{field} differs across curves: {vals}")
        else:
            findings.append(f"[ok] config.{field} identical across all curves: {ref['config'][field]}")
    for field in PROV_INVARIANTS:
        vals = {label: by_conc[min(by_conc)][0]["prov"][field] for label, by_conc in data.items()}
        if len(set(vals.values())) != 1:
            findings.append(f"**CONTROL BROKEN** prov.{field} differs across curves: {vals}")
        else:
            findings.append(f"[ok] prov.{field} identical across all curves: {ref['prov'][field][:12]}…")

    # git_sha is EXPECTED to differ -- code changed between sessions. Report the
    # values so the Method section can state which SHA produced which curve,
    # rather than leaving a reader to assume one build produced all four.
    shas = {label: by_conc[min(by_conc)][0]["prov"]["git_sha"][:10] for label, by_conc in data.items()}
    findings.append(f"[info] git_sha per curve (expected to differ): {shas}")

    return findings


def agg(records: list[dict], path: list[str]) -> dict[str, float]:
    """Mean/min/max/spread of a nested numeric field across repeats."""
    vals = []
    for r in records:
        v = r
        for k in path:
            v = v[k]
        vals.append(v)
    mean = statistics.fmean(vals)
    return {
        "mean": mean,
        "min": min(vals),
        "max": max(vals),
        "spread_pct": (max(vals) - min(vals)) / mean * 100 if mean else 0.0,
        "n": len(vals),
    }


TOK = ["throughput", "output_tok_s"]
GOOD = ["throughput", "goodput_req_s"]
TTFT50 = ["latency", "ttft_ms", "p50"]
TTFT95 = ["latency", "ttft_ms", "p95"]
TPOT50 = ["latency", "tpot_ms", "p50"]


def worst_spread(data) -> tuple[float, str, int]:
    worst, where, lvl = 0.0, "", 0
    for label, by_conc in data.items():
        for c, recs in by_conc.items():
            s = agg(recs, TOK)["spread_pct"]
            if s > worst:
                worst, where, lvl = s, label, c
    return worst, where, lvl


def verdict(data, levels) -> tuple[str, list[str]]:
    """Apply SCOPE.md's pre-registered refutation criterion. No judgement calls.

    > Refuted if one configuration leads at every concurrency level on BOTH
    > cards, OR the ranking changes at the same concurrency on both.
    """
    notes = []
    counterexamples = []
    for card, off, on in (("A100", "A100 off", "A100 on"), ("L4", "L4 off", "L4 on")):
        for c in levels:
            if agg(data[on][c], TOK)["mean"] <= agg(data[off][c], TOK)["mean"]:
                counterexamples.append((card, c))
    if not counterexamples:
        notes.append(
            "Prefix caching leads at every concurrency level on both cards "
            f"({2 * len(levels)} comparisons, 0 counter-examples). No ranking "
            "change exists, so no ranking-change point can move with hardware."
        )
        return "REFUTED", notes
    notes.append(f"ranking changes at: {counterexamples}")
    return "NOT REFUTED BY CLAUSE 1 — inspect where the ranking changes per card", notes


def cost_table(data, levels, price: dict[str, float]):
    rows = []
    for c in levels:
        row = {"conc": c}
        for label, _f, card, _pc in CURVES:
            row[label] = agg(data[label][c], TOK)["mean"] / price[card]
        row["off_ratio"] = row["L4 off"] / row["A100 off"]
        row["on_ratio"] = row["L4 on"] / row["A100 on"]
        rows.append(row)
    return rows


def price_sensitivity(data, levels, price, frac):
    """Does the cost ordering survive the prices moving +/- frac?

    Worst case for the L4 leading: L4 priced high, A100 priced low.
    """
    worst = {"A100": price["A100"] * (1 - frac), "L4": price["L4"] * (1 + frac)}
    best = {"A100": price["A100"] * (1 + frac), "L4": price["L4"] * (1 - frac)}
    out = []
    for name, p in (("worst case for L4", worst), ("best case for L4", best)):
        rows = cost_table(data, levels, p)
        l4_leads_off = all(r["off_ratio"] > 1 for r in rows)
        l4_leads_on = all(r["on_ratio"] > 1 for r in rows)
        out.append((name, p, l4_leads_off, l4_leads_on,
                    min(r["off_ratio"] for r in rows), min(r["on_ratio"] for r in rows)))
    return out


def breakeven(data, levels, price):
    """The A100 price at which the two cards tie on tokens-per-dollar.

    Reported instead of a single ratio because it is the number a reader can
    check against their own contract: below this A100 price, the A100 leads.
    """
    out = []
    for suffix in ("off", "on"):
        ratios = []
        for c in levels:
            a = agg(data[f"A100 {suffix}"][c], TOK)["mean"]
            l = agg(data[f"L4 {suffix}"][c], TOK)["mean"]
            # A100 leads when a/price_a100 > l/price_l4  =>  price_a100 < a*price_l4/l
            ratios.append(a * price["L4"] / l)
        out.append((suffix, min(ratios), max(ratios)))
    return out


# --------------------------------------------------------------------------
# Output
# --------------------------------------------------------------------------

def fmt_md_table(header: list[str], rows: list[list[str]], align: str = "r") -> str:
    sep = "|" + "|".join(("---:" if align == "r" else "---") for _ in header) + "|"
    out = ["| " + " | ".join(header) + " |", sep]
    for r in rows:
        out.append("| " + " | ".join(r) + " |")
    return "\n".join(out)


def report(data, levels, price, sens_frac, emit):
    md = emit == "markdown"
    P = print

    if not md:
        P("=" * 78)
        P("MEASUREMENT 03 -- ANALYSIS")
        P("=" * 78)
        P("\n--- INTEGRITY ---")
        for f in check_integrity(data):
            P("  " + f)
    else:
        bad = [f for f in check_integrity(data) if f.startswith("**")]
        if bad:
            P("> **INTEGRITY FAILURES**\n>\n" + "\n".join("> - " + b for b in bad) + "\n")

    # ---- Throughput + payoff ratio ----
    rows = []
    for c in levels:
        cells = [str(c)]
        for label, _f, _card, _pc in CURVES:
            a = agg(data[label][c], TOK)
            cells.append(f"{a['mean']:.1f}")
        ra = agg(data["A100 on"][c], TOK)["mean"] / agg(data["A100 off"][c], TOK)["mean"]
        rl = agg(data["L4 on"][c], TOK)["mean"] / agg(data["L4 off"][c], TOK)["mean"]
        cells += [f"{ra:.2f}x", f"{rl:.2f}x"]
        rows.append(cells)
    hdr = ["conc", "A100 off", "A100 on", "L4 off", "L4 on", "A100 on/off", "L4 on/off"]
    P("\n### Throughput (output tokens/sec) and prefix-caching payoff\n" if md else "\n--- THROUGHPUT tok/s ---")
    P(fmt_md_table(hdr, rows) if md else "\n".join("  " + " ".join(f"{x:>11}" for x in r) for r in [hdr] + rows))

    # ---- Spread ----
    w, where, lvl = worst_spread(data)
    conc1 = {label: agg(data[label][1], TOK)["spread_pct"] for label in data}
    line = (f"Worst run-to-run spread anywhere in the grid: **{w:.2f}%** "
            f"({where}, concurrency {lvl}). At concurrency 1 the spreads are "
            + ", ".join(f"{k} {v:.3f}%" for k, v in conc1.items()) + ".")
    P("\n" + line if md else f"\n--- SPREAD ---\n  {line}")

    # ---- Verdict ----
    v, notes = verdict(data, levels)
    P(f"\n### Verdict against the pre-registered criterion\n\n**{v}**\n" if md else f"\n--- VERDICT: {v} ---")
    for n in notes:
        P(("- " + n) if md else "  " + n)

    # ---- Latency + goodput ----
    rows = []
    for c in levels:
        cells = [str(c)]
        for label, _f, _card, _pc in CURVES:
            cells.append(f"{agg(data[label][c], TPOT50)['mean']:.2f}")
        for label, _f, _card, _pc in CURVES:
            cells.append(f"{agg(data[label][c], GOOD)['mean']:.2f}")
        rows.append(cells)
    hdr = ["conc"] + [f"TPOT {l}" for l, *_ in CURVES] + [f"good {l}" for l, *_ in CURVES]
    P("\n### TPOT p50 (ms) and goodput (req/s), SLO TTFT<1000ms and TPOT<50ms\n" if md
      else "\n--- TPOT p50 ms | GOODPUT req/s ---")
    P(fmt_md_table(hdr, rows) if md else "\n".join("  " + " ".join(f"{x:>10}" for x in r) for r in [hdr] + rows))

    # ---- Cost ----
    P(f"\n### Cost-normalised throughput (tokens/sec per $-hour)\n" if md else "\n--- COST ---")
    P(f"Prices used: A100 ${price['A100']:.4f}/hr, L4 ${price['L4']:.4f}/hr. "
      "**Not measured** -- record vendor, tier and date in Method.\n")
    rows = []
    for r in cost_table(data, levels, price):
        rows.append([str(r["conc"]), f"{r['A100 off']:.1f}", f"{r['L4 off']:.1f}", f"{r['off_ratio']:.2f}x",
                     f"{r['A100 on']:.1f}", f"{r['L4 on']:.1f}", f"{r['on_ratio']:.2f}x"])
    hdr = ["conc", "A100 off", "L4 off", "L4/A100", "A100 on", "L4 on", "L4/A100"]
    P(fmt_md_table(hdr, rows) if md else "\n".join("  " + " ".join(f"{x:>10}" for x in r) for r in [hdr] + rows))

    # ---- Sensitivity ----
    P(f"\n### Price sensitivity (+/-{sens_frac:.0%})\n" if md else f"\n--- PRICE SENSITIVITY +/-{sens_frac:.0%} ---")
    for name, p, off_ok, on_ok, min_off, min_on in price_sensitivity(data, levels, price, sens_frac):
        P(f"- **{name}** (A100 ${p['A100']:.2f}, L4 ${p['L4']:.2f}): "
          f"L4 leads at every level — caching-off {'YES' if off_ok else 'NO'} "
          f"(min ratio {min_off:.2f}x), caching-on {'YES' if on_ok else 'NO'} "
          f"(min ratio {min_on:.2f}x)")

    P("\n**Break-even A100 price** — the hourly rate at which the cards tie on "
      f"tokens-per-dollar, holding L4 at ${price['L4']:.2f}/hr:\n" if md else "\n--- BREAK-EVEN ---")
    for suffix, lo, hi in breakeven(data, levels, price):
        P(f"- caching {suffix}: ${lo:.2f}–${hi:.2f}/hr across the concurrency range. "
          f"Below that the A100 leads on cost; above it the L4 does.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", type=Path, default=ROOT / "results")
    ap.add_argument("--price-a100", type=float, default=3.67,
                    help="USD/hour. NOT measured -- see Method. Default is a placeholder.")
    ap.add_argument("--price-l4", type=float, default=0.70,
                    help="USD/hour. NOT measured -- see Method. Default is a placeholder.")
    ap.add_argument("--sensitivity", type=float, default=0.30,
                    help="fractional price swing tested in both directions")
    ap.add_argument("--emit", choices=["report", "markdown"], default="report")
    args = ap.parse_args()

    data = load(args.results)
    levels = sorted(next(iter(data.values())))
    price = {"A100": args.price_a100, "L4": args.price_l4}

    broken = [f for f in check_integrity(data) if f.startswith("**")]
    if broken:
        print("REFUSING: controls are not identical across curves.", file=sys.stderr)
        for b in broken:
            print("  " + b, file=sys.stderr)
        raise SystemExit(2)

    report(data, levels, price, args.sensitivity, args.emit)


if __name__ == "__main__":
    main()
