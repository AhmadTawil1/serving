# SCOPE — Measurement 03: Serving

**Committed 5 Aug 2026 (Day 1 of this block — M03-SERVING.md's own calendar
labels this "Day 18"; that numbering is the programme's original 28-day
schedule and is kept in section headers there for cross-reference, but the
block actually starts today, immediately after Measurement 02's halt.
`Ingestion/LOG.md` → *Day 3 — Measurement 02 halted* names Serving as next,
explicitly because its ground truth is machine-generated and the annotation
failure that stopped M02 cannot recur here.**

## In scope

One serving harness driving vLLM and TensorRT-LLM behind identical
OpenAI-compatible endpoints, serving one pinned model
(`Qwen/Qwen2.5-7B-Instruct`, revision `a09a3545…`, `configs/model_pins.yaml`),
swept over framework × card × concurrency (8 log-spaced levels, 1–128) at
FP16 — the main sweep, 96 measured runs — plus a precision ladder (FP8/INT8/
INT4 where the architecture permits) at three concurrency levels (1, 16,
128) on both an **A100-SXM4-40GB** and an **L4-24GB**. Ground truth is
machine-generated: TTFT, TPOT and throughput are read off a clock, and each
precision's output is gated against the FP16 reference on the *same card and
framework* before its throughput is allowed into the figure (§4.4). Output:
four cost-normalised throughput curves (tokens/sec per GPU-dollar-hour vs.
concurrency) and the crossover concurrency per card, with an interval.

## Out of scope

Kernel-level quantization research — that is APEX4/AnyBCQ territory
(`REFERENCE.md` §4.3) and this study is deliberately not competing there, it
is measuring framework-level behaviour under load. Any hardware beyond the
A100/L4 pair (the T4 is SM75 and TensorRT-LLM's INT8/INT4/fused-attention
paths reject it, `REFERENCE.md` §3). Open-loop (fixed arrival-rate) load
patterns — closed-loop only (§4.2). Any input/output token length other than
the frozen 512/128 pair (`serving/config.py`). Multi-adapter or LoRA serving.
Generation quality as anything other than a pass/fail gate per precision —
it is a guard here, not an axis (§2.2).

## Hypothesis 3 — committed before any sweep

> vLLM leads at low concurrency; TensorRT-LLM leads above a crossover point;
> and **the crossover moves with GPU tier**, arriving at a different
> concurrency on the L4 than on the A100, because compiled engine-level
> optimisation and memory-efficient batching are rewarded differently by a
> card with less bandwidth and less headroom.

**Direction fixed today: vLLM-low / TensorRT-LLM-high** — the option
`M03-SERVING.md` §1.1 lists first and `REFERENCE.md` §7 already commits to.
Reasoning: at concurrency 1 there is nothing to batch and decode is
bandwidth-bound, so neither framework can compile its way past the memory
wall — they should be close. TensorRT-LLM's fused kernels and in-flight
batching are load-time optimisations, and they show up under load, not at
concurrency 1. The alternative (TRT-low / vLLM-high, on the theory that
compiled kernels shorten every request regardless of load while
PagedAttention's memory-packing advantage only pays off under concurrency)
was equally arguable a priori and is **not** being pre-registered — per
§1.1, only one direction can be committed, and it is not revised after
seeing data. If the measurement contradicts this direction, that is a result
and gets reported as one.

**Refuted if.** One framework leads at every concurrency level on **both**
cards, **or** the crossover sits at the same concurrency on both.

**Partial if.** A crossover exists and moves, but the movement is within
run-to-run variance (3 repeats), or it moves for a mechanism other than the
predicted one — e.g. an OOM cliff rather than a gradual trade. Report which,
and say plainly the mechanism is not the one predicted.

**Never cut.** Both cards, and the concurrency axis. A single-card sweep is
a vendor blog post; a single-concurrency comparison is what the literature
already has. Cut the precision ladder first (§1's table), then the upper
concurrency rungs.

## The Day 21 engine hard stop

If both TensorRT-LLM engines (A100 and L4) are not serving by end of the
block's Day 4 (labelled "Day 21" in `M03-SERVING.md`'s programme-relative
numbering), the TensorRT-LLM axis is dropped and the `M03-SERVING.md` §17
contingency executes: report the A100 crossover if that engine landed, with
the L4 build failure reported as a portability result in its own right, plus
a vLLM-only L4 concurrency curve so the card axis still has data. Fixed
today, before it is inconvenient.

## What this is, and is not

This is a **framework-and-hardware** study: does the vLLM/TensorRT-LLM
ranking invert with load, and does where it inverts depend on the card. It
is not a quantization-accuracy study (quality is a gate, §2.2) and not a
kernel-level efficiency study (§4.3's gap sentence is about the framework
layer, not the kernel layer). Engine build time and precision refusals are
recorded as first-class results, not overhead or noise — §6, §2.3.

*Verbatim from `M03-SERVING.md` §1 and `REFERENCE.md` §7. Where this file
and those disagree on scope, this file wins as of 5 Aug 2026.*
