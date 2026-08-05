# SCOPE — Measurement 03: Serving

**Rewritten 5 Aug 2026, after the TensorRT-LLM spike refused and before any
further measurement.** The original scope compared two serving *frameworks*.
TensorRT-LLM 1.2.1 installs on the target runtime but cannot load its own
bindings — it requires `libcublasLt.so.13`, which is absent and is not
obtainable from PyPI or from NVIDIA's own index (`LOG.md`, Day 7). The
pre-registered hard-stop rule below fired, and §17's named contingency is now
the study.

*This is the second scope change in the programme, and like Measurement 02's it
is recorded rather than quietly absorbed. The difference is that this one is
forced by a toolchain refusal that is itself a reportable result, and the
thesis question survives the swap intact.*

---

## In scope

One serving harness driving **vLLM** behind an OpenAI-compatible endpoint,
serving one pinned model (`Qwen/Qwen2.5-7B-Instruct`, revision `a09a3545…`,
`configs/model_pins.yaml`) at FP16, swept over **configuration × card ×
concurrency**:

| Axis | Levels |
|---|---|
| **Prefix caching** | on / off |
| **Chunked prefill** | on / off |
| **max-num-seqs** | scheduler batch cap, two levels |
| **Card** | A100-SXM4-40GB · L4-24GB |
| **Concurrency** | 1 · 2 · 4 · 8 · 16 · 32 · 64 · 128 |

Ground truth is machine-generated: TTFT, TPOT and throughput read off a clock,
client-side, closed-loop (§4.1, §4.2). Output: cost-normalised throughput and
goodput against concurrency, per configuration, per card — and **the
concurrency at which the configuration ranking changes, on each card.**

**Already measured** (A100, prefix caching on and off, 8 concurrency levels ×
3 repeats each, `results/`): two complete curves. The L4 pass is what remains.

## Out of scope

**TensorRT-LLM**, and framework comparison generally — dropped by the
hard-stop rule, with the refusal reported as a result rather than a gap
(`LOG.md` Day 7, §4). Kernel-level quantization research — APEX4/AnyBCQ
territory (`REFERENCE.md` §4.3), deliberately not competed with. Any hardware
beyond the A100/L4 pair. Open-loop load patterns — closed-loop only. Any
input/output token length other than the frozen 512/128 pair
(`serving/config.py`). Multi-adapter or LoRA serving. Generation quality as
anything other than a pass/fail gate. **Engine build time and precision
refusals**, which were TensorRT-LLM-specific and no longer apply.

The **precision ladder** (FP8/INT8/INT4) is retained as optional and remains
the first thing to cut. FP16 across the configuration axis is the study.

## Hypothesis 3 — restated 5 Aug 2026, before any L4 measurement

> The optimal vLLM configuration is **concurrency-dependent** — no single
> configuration leads across the whole load range — and **the concurrency at
> which the ranking changes depends on the card**, arriving at a different
> point on the L4 than on the A100 because a card with less memory bandwidth
> and less KV capacity is rewarded differently by the same setting.

**Direction retained from the original pre-registration**, with the same
reasoning transposed: settings whose benefit is memory-efficiency-shaped
(prefix caching, scheduler batching) should be worth little at concurrency 1,
where there is nothing to batch and decode is bandwidth-bound, and worth
increasingly more under load. This is now partly *observed* rather than
predicted — the A100 prefix-caching contrast rises monotonically from 1.01× at
concurrency 1 to 2.56× at 128 (`LOG.md` Day 6). **The untested half is whether
that curve has the same shape on the L4, and that is the live prediction.**

**Refuted if.** One configuration leads at every concurrency level on **both**
cards, **or** the ranking changes at the same concurrency on both.

**Partial if.** The ranking changes and the change point moves, but the
movement is inside run-to-run variance (3 repeats; observed spread ≤0.38% on
the A100), or it moves for a mechanism other than the predicted one — an OOM
cliff rather than a gradual trade. Report which, and say plainly the mechanism
is not the one predicted.

**Never cut.** Both cards, and the concurrency axis. A single-card
configuration sweep is a tuning guide, not a hardware-dependence result. Cut
`max-num-seqs` first, then chunked prefill. **Prefix caching on/off is the
lead knob and is not cut** — it is the one already shown to have a
concurrency-dependent payoff.

## What the hard stop was, and that it fired

Written Day 1, applied Day 7 without modification:

> **Neither engine serving.** The TensorRT-LLM axis is dropped **entirely**,
> and the Day 18 contingency executes instead (`M03-SERVING.md` §17,
> "TensorRT-LLM will not install on the runtime"): the framework axis becomes
> vLLM configuration knobs. This file gets rewritten then, not later.

No engine build was attempted on either card, because the failure occurred at
import, before any build. **No build-time data exists and none is claimed** —
the §4 build-time table is removed rather than left with empty rows.

## Held fixed, and reported as configuration

Unchanged from the original scope except where the framework axis touched them:

| Held | Value |
|---|---|
| Model | `Qwen/Qwen2.5-7B-Instruct`, revision `a09a3545…`, pinned via `pins.py` |
| Precision | FP16 (`--dtype float16`; the model is natively bf16 and is cast) |
| Input / output tokens | 512 / 128, frozen, `prompts_sha 3b3efb02` |
| Max model length | 2048 |
| Sampling | Greedy, `temperature=0` |
| Request pattern | Closed-loop, N workers, one in flight each |
| Warmup | 32 requests at the target concurrency, discarded |
| Repeats | 3, one record written per repeat |

**Prefix caching moves from held-fixed control to swept knob.** It was fixed
off on Day 6 precisely to make the vLLM/TensorRT-LLM comparison valid; with
that comparison gone, the setting becomes the study's most informative axis.

## What this is, and is not

This is a **configuration-and-hardware** study: does the best vLLM setting
depend on how loaded the server is, and does the answer depend on the card. It
is not a framework comparison — that was attempted and is reported as refused
at the toolchain level, with the error strings verbatim. It is not a
quantization-accuracy study, and not a kernel-level efficiency study.

The programme-level claim is unchanged by any of this. It was never *which
framework wins*; it is **that the answer depends on the hardware**, and the
configuration axis tests exactly that with two cards and a concurrency sweep.

*Supersedes the 5 Aug (morning) version of this file. Where this file and
`M03-SERVING.md` or `REFERENCE.md` §7 disagree, this file wins as of 5 Aug
2026. `LOG.md` Day 7 records why.*
