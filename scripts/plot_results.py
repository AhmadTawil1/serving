"""Figures for Paper 3. Every value comes from `analysis.py`'s loaders, so a
figure and a table can never disagree.

    python scripts/plot_results.py
    python scripts/plot_results.py --price-a100 3.67 --price-l4 0.70

Four figures:

  fig1_throughput.pdf   throughput vs concurrency, four curves
  fig2_payoff.pdf       prefix-caching payoff ratio vs concurrency, per card
                        -- THE headline: the two lines lying on top of each
                        other is the transfer result
  fig3_slo.pdf          TPOT p50 against the SLO ceiling, and goodput
                        -- why the L4 is unusable despite the payoff transferring
  fig4_cost.pdf         cost-normalised throughput, with the break-even A100
                        price marked, because the ordering is NOT robust to the
                        price assumption (LOG.md Day 8 sensitivity)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from analysis import CURVES, GOOD, TOK, TPOT50, agg, breakeven, load  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent

# A100 blue, L4 amber; caching-off dashed, caching-on solid.
STYLE = {
    "A100 off": dict(color="#1D4E6B", ls="--", marker="o", ms=4),
    "A100 on": dict(color="#1D4E6B", ls="-", marker="o", ms=4),
    "L4 off": dict(color="#9C6216", ls="--", marker="s", ms=4),
    "L4 on": dict(color="#9C6216", ls="-", marker="s", ms=4),
}


def _finish(ax, levels, xlabel="concurrency (closed-loop, log scale)"):
    ax.set_xscale("log", base=2)
    ax.set_xticks(levels)
    ax.set_xticklabels([str(c) for c in levels])
    ax.set_xlabel(xlabel)
    ax.grid(alpha=0.25, lw=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def fig_throughput(data, levels, out):
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for label, *_ in CURVES:
        y = [agg(data[label][c], TOK)["mean"] for c in levels]
        ax.plot(levels, y, label=label, **STYLE[label])
    ax.set_yscale("log")
    ax.set_ylabel("output tokens/sec")
    ax.set_title("Throughput vs concurrency — vLLM, Qwen2.5-7B-Instruct, FP16", fontsize=10)
    _finish(ax, levels)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def fig_payoff(data, levels, out):
    """The transfer result. Two lines that nearly coincide across a ~4.4-5.3x
    hardware gap."""
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for card, off, on, colour, mk in (
        ("A100", "A100 off", "A100 on", "#1D4E6B", "o"),
        ("L4", "L4 off", "L4 on", "#9C6216", "s"),
    ):
        y = [agg(data[on][c], TOK)["mean"] / agg(data[off][c], TOK)["mean"] for c in levels]
        ax.plot(levels, y, color=colour, marker=mk, ms=4.5, lw=1.8, label=card)
        ax.annotate(f"{y[-1]:.2f}x", (levels[-1], y[-1]), textcoords="offset points",
                    xytext=(6, -2), fontsize=8, color=colour)
    ax.axhline(1.0, color="#848C83", lw=0.8, ls=":")
    ax.text(levels[0], 1.02, "no benefit", fontsize=7.5, color="#848C83")
    ax.set_ylabel("throughput with prefix caching ÷ without")
    ax.set_title("The payoff from prefix caching transfers across hardware tiers", fontsize=10)
    _finish(ax, levels)
    ax.legend(frameon=False, fontsize=8, title="card", title_fontsize=8)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def fig_slo(data, levels, out, tpot_slo=50.0):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.6, 3.9))
    for label, *_ in CURVES:
        ax1.plot(levels, [agg(data[label][c], TPOT50)["mean"] for c in levels],
                 label=label, **STYLE[label])
    ax1.axhline(tpot_slo, color="#8C2F2F", lw=1.1, ls="-")
    ax1.text(levels[0], tpot_slo * 1.06, f"SLO ceiling {tpot_slo:.0f} ms",
             fontsize=7.5, color="#8C2F2F")
    ax1.set_yscale("log")
    ax1.set_ylabel("TPOT p50 (ms)")
    ax1.set_title("Per-token latency against the SLO", fontsize=10)
    _finish(ax1, levels)
    ax1.legend(frameon=False, fontsize=7.5)

    for label, *_ in CURVES:
        ax2.plot(levels, [agg(data[label][c], GOOD)["mean"] for c in levels],
                 label=label, **STYLE[label])
    ax2.set_ylabel("goodput (req/s meeting the SLO)")
    ax2.set_title("Both L4 curves sit flat on zero", fontsize=10)
    _finish(ax2, levels)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def fig_cost(data, levels, price, out):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.6, 3.9),
                                   gridspec_kw={"width_ratios": [1.4, 1]})
    for label, _f, card, _pc in CURVES:
        y = [agg(data[label][c], TOK)["mean"] / price[card] for c in levels]
        ax1.plot(levels, y, label=label, **STYLE[label])
    ax1.set_yscale("log")
    ax1.set_ylabel("output tokens/sec per $-hour")
    ax1.set_title(f"Cost-normalised (A100 ${price['A100']:.2f}/hr, L4 ${price['L4']:.2f}/hr)",
                  fontsize=9.5)
    _finish(ax1, levels)
    ax1.legend(frameon=False, fontsize=7.5)

    # Break-even panel: the A100 price at which the cards tie, per level.
    for suffix, colour, mk in (("off", "#1D4E6B", "o"), ("on", "#9C6216", "s")):
        be = [agg(data[f"A100 {suffix}"][c], TOK)["mean"] * price["L4"]
              / agg(data[f"L4 {suffix}"][c], TOK)["mean"] for c in levels]
        ax2.plot(levels, be, color=colour, marker=mk, ms=4, lw=1.6,
                 label=f"caching {suffix}")
    ax2.axhline(price["A100"], color="#8C2F2F", lw=1.1)
    ax2.text(levels[0], price["A100"] * 1.01,
             f"assumed A100 price ${price['A100']:.2f}", fontsize=7.5, color="#8C2F2F")
    ax2.set_ylabel("break-even A100 price ($/hr)")
    ax2.set_title("Above the line the L4 is cheaper per token; below it the A100 is",
                  fontsize=8.5)
    _finish(ax2, levels)
    ax2.legend(frameon=False, fontsize=7.5)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", type=Path, default=ROOT / "results")
    ap.add_argument("--figs", type=Path, default=ROOT / "figs")
    ap.add_argument("--price-a100", type=float, default=3.67)
    ap.add_argument("--price-l4", type=float, default=0.70)
    args = ap.parse_args()

    data = load(args.results)
    levels = sorted(next(iter(data.values())))
    price = {"A100": args.price_a100, "L4": args.price_l4}
    args.figs.mkdir(parents=True, exist_ok=True)

    fig_throughput(data, levels, args.figs / "fig1_throughput.pdf")
    fig_payoff(data, levels, args.figs / "fig2_payoff.pdf")
    fig_slo(data, levels, args.figs / "fig3_slo.pdf")
    fig_cost(data, levels, price, args.figs / "fig4_cost.pdf")


    print(f"wrote 4 figures to {args.figs}")
    for suffix, lo, hi in breakeven(data, levels, price):
        print(f"  break-even A100 price, caching {suffix}: ${lo:.2f}-${hi:.2f}/hr")


if __name__ == "__main__":
    main()
