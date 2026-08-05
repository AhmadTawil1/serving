# Measurement 03 — Serving
### Day-by-day working plan · what to code, what to write

**Dates:** Tue 18 Aug → Tue 25 Aug 2026 · 7 measure + 1 write
**Paper:** *Does the serving-framework ranking invert with concurrency, and does the inversion point move with GPU tier?*
**Status:** not started

Execution plan for Block 3. Policy lives in `REFERENCE.md`; this file holds the
daily detail and the decisions taken while planning it. Where the two disagree,
`REFERENCE.md` wins. Companion to `research/m03-serving-plan.html`.

**This is the last measurement of the programme, and it now carries more weight
than planned.** Measurement 02 was halted at the annotation step on 5 Aug
(`Ingestion/LOG.md`, Day 3). Paper 4's hardware axis rests on Measurements 01
and 03 alone, so an M03 that produces two clean curves per card matters more
than an M03 that attempts a wider sweep and finishes half of it.

---

## 1. Hypothesis 3 — committed before any sweep

> vLLM leads at low concurrency; TensorRT-LLM leads above a crossover point;
> and **the crossover moves with GPU tier**, arriving at a different concurrency
> on the L4 than on the A100, because compiled engine-level optimisation and
> memory-efficient batching are rewarded differently by a card with less
> bandwidth and less headroom.

| | |
|---|---|
| **Refuted if** | One framework leads at every concurrency level on **both** cards, **or** the crossover sits at the same concurrency on both. |
| **Partial if** | A crossover exists and moves, but the movement is within run-to-run variance, or it moves for a mechanism other than the predicted one (e.g. an OOM cliff rather than a gradual trade). Report which, and say plainly the mechanism is not the one predicted. |
| **Never cut** | Both cards, and the concurrency axis. A single-card sweep is a vendor blog post; a single-concurrency comparison is what the literature already has. Cut the precision ladder first, then the upper concurrency rungs. |
| **Write it first** | Into `SCOPE.md` and Paper 3 §1 on Tue 18 Aug, before a single request is sent. |

### 1.1 The direction is an open decision, not a detail

**Flagged 5 Aug 2026.** `REFERENCE.md` §7 states vLLM low / TensorRT-LLM high.
The opposite ordering is equally arguable:

- **For vLLM-low / TRT-high** (as committed): at concurrency 1 there is nothing
  to batch and decode is bandwidth-bound, so neither framework can compile its
  way past the memory wall — they should be close. TensorRT-LLM's fused kernels
  and in-flight batching are load-time optimisations, and they show up under
  load.
- **For TRT-low / vLLM-high:** TensorRT-LLM's compiled kernels shorten *every*
  request, whether there is one or fifty, so it should lead at concurrency 1.
  vLLM's PagedAttention is a memory-packing advantage worth nothing at
  concurrency 1 and worth a great deal at concurrency 128.

**Both cannot be pre-registered.** Pick one on Day 18, write it into `SCOPE.md`,
and do not revise it after seeing data. If the measurement contradicts the
committed direction, that is a result and it gets reported as one — a
pre-registration that is never wrong was never a prediction.

The programme-level claim survives either way, because the claim is not *which
framework wins*. It is **that the answer depends on the card**. That is what
`Refuted if` is written against.

### 1.2 Headline figure

Output tokens/sec **per GPU-dollar-hour** versus concurrency, log x-axis. Four
curves: two frameworks × two cards. The prediction is **two crossings at
different x positions**, not one. Mark each crossing with a vertical rule and
label the concurrency.

Cost-normalising is what makes the two cards comparable at all. Raw throughput
puts the A100 above the L4 everywhere and the figure says nothing. Divided by
price, the L4 can win outright — and if it does, that is the most quotable
sentence in the paper.

---

## 2. Four things that differ from Measurements 01 and 02

### 2.1 Ground truth is machine-generated

M01 needed 30 hand-labelled character spans. M02 needed 20 hand-labelled time
intervals and **stopped because they were never written**. M03 needs neither.
TTFT, TPOT and throughput are read off a clock; the load generator produces its
own ground truth.

This is the reason M03 follows the M02 halt rather than a retry. The failure
mode that ended the previous block cannot occur in this one. Say so in §5
Threats — an honest account of why the third study is shaped the way it is.

**What replaces the annotation risk is the toolchain risk.** TensorRT-LLM engine
builds are the M03 equivalent of the M02 labelling grind: slow, uninformative
when they fail, and budgeted a full day per card. §6 bounds it.

### 2.2 Quality is a guard, not an axis

In M01 recall was the y-axis. In M02 it would have been. Here, **quality is a
pass/fail gate on each precision level**, not a reported trade-off curve.

The reason: FP16 and FP8 and INT4 of the same model are not supposed to produce
different answers, and any throughput number bought with degraded output is
fraudulent. So each precision gets a fixed quality check before its throughput
is allowed into the figure. Fail the check, and the precision is reported as
*rejected on quality* — which is itself a finding, and a more useful one than a
fast number nobody can trust.

### 2.3 Both cards are mandatory, and the precision support differs between them

In M01 the L4 pass was the last thing to cut. **Here it cannot be cut**, because
after M02's halt there is no other hardware comparison left in the programme.

There is also an asymmetry worth planning around: **A100 is Ampere (SM80) and
does not have FP8 tensor cores; L4 is Ada (SM89) and does.** Verify this on the
hardware rather than taking it from here — but if it holds, the *cheaper* card
supports a precision the expensive one refuses, which is the practical half of
the hardware-dependence thesis stated in one line.

**Record every refusal with its exact error.** Which precisions each card
rejects is a result, not an obstacle (`REFERENCE.md` §7).

### 2.4 The load generator is part of the system under test

M01 and M02 measured a pipeline the harness drove synchronously. Here the
harness is a concurrent client, and a slow client silently caps the server.

If the load generator cannot sustain 128 in-flight requests, the measurement at
concurrency 128 is a measurement of the client. **This must be ruled out
explicitly on Day 19**, not assumed — see §4.2.

---

## 3. The sweep — frozen 18 Aug

| Axis | Levels | Notes |
|---|---|---|
| **Framework** | vLLM · TensorRT-LLM | The comparison |
| **Card** | A100 40GB · L4 24GB | The hypothesis |
| **Concurrency** | 1 · 2 · 4 · 8 · 16 · 32 · 64 · 128 | Log-spaced; the crossover is located on this axis |
| **Precision** | FP16 · FP8 · INT8 · INT4 | Where the architecture permits. Refusals recorded |

### Held fixed and reported as configuration

These are the control, and every one of them confounds the result if it moves:

| Held | Value | Why it matters |
|---|---|---|
| Model | One model, one revision, pinned via `pins.py` | Obvious, and the easiest thing to get wrong across two frameworks |
| Input length | Fixed token count, identical prompt set | Prefill cost scales with it; a longer prompt on one framework is a different experiment |
| Output length | Fixed, forced with `ignore_eos` / `min_tokens` | **Critical.** If one framework's replies are shorter, its TPOT looks better for free |
| Sampling | Greedy (`temperature=0`) | Removes a variance source and makes the quality check deterministic |
| Max model length | Same on both | Changes KV cache allocation, which changes achievable concurrency |
| Request pattern | Closed-loop, N workers, each 1 in flight | §4.2 |
| Warmup | Fixed request count, discarded | §4.3 |

### Run counts

Main sweep: 2 frameworks × 2 cards × 8 concurrency levels = **32 curve points**,
each 3 repeats = **96 measured runs**, all at FP16.

Precision ladder: 4 precisions × 2 frameworks × 2 cards, at **three**
concurrency levels only (1, 16, 128) = 48 points × 3 repeats = 144 runs, minus
whatever the cards refuse.

**The precision ladder is the first thing to cut.** The main sweep is the
hypothesis; the ladder is supporting evidence. If Day 22 is going badly, run
FP16 and INT4 only and say why.

---

## 4. Five decisions that are easy to get wrong

### 4.1 TTFT and TPOT are client-side measurements on a stream

Do not take them from the server's own logs. The server does not know when the
first byte reached the client, and the framing question — *when does text start
appearing for a user* — is a client-side question.

```
request sent            → t0
first streamed chunk    → t1        TTFT   = t1 - t0
last streamed chunk     → t2        TPOT   = (t2 - t1) / (n_output_tokens - 1)
```

Note the `- 1`: the first token is already charged to TTFT, and counting it
twice makes TPOT look better on every framework equally, which is the kind of
error that survives review because it cancels.

Both frameworks expose OpenAI-compatible streaming endpoints, so **one client
drives both**. That is not a convenience — a separate client per framework is a
confound, and it is the single most likely way this study produces a difference
that is really a harness artifact.

### 4.2 Concurrency must be closed-loop, and the client must be proven not to be the bottleneck

**Closed-loop:** N workers, each sends a request, waits for the full response,
sends the next. In-flight count is exactly N at all times. That is what
"concurrency N" means here, and it is what makes the x-axis a real axis.

Open-loop (fixed arrival rate) is the other valid design, but it produces
queue-collapse behaviour above saturation and the crossover gets buried in
queueing delay rather than framework behaviour.

**Prove the client is not the limit** on Day 19, before any real number:

- Run the sweep against a **mock server** that returns a canned stream with a
  fixed sleep. If measured throughput at concurrency 128 does not scale to the
  mock's theoretical ceiling, the client is the bottleneck.
- Use `asyncio` + `aiohttp`, not threads. 128 threads on a Colab CPU will not
  hold 128 streams cleanly.
- Record client-side CPU utilisation in every run record. A pegged client CPU
  invalidates the point.

### 4.3 Warmup is not optional, and the first requests are engine artifacts

First requests trigger CUDA graph capture, lazy kernel compilation, memory pool
growth and cache warming. On TensorRT-LLM the first request after engine load
can be seconds slower than steady state.

- Fixed warmup: **N discarded requests at the target concurrency**, not at
  concurrency 1, and not a fixed number of seconds.
- Then measure a fixed window.
- Record warmup count in the run record so the number is reconstructable.
- Restart the server between *precision* changes. Do not reuse a process across
  precisions and trust it to have released memory.

### 4.4 Quality is checked per precision, or throughput is bought silently

Before a precision's throughput enters the figure:

- A fixed prompt set, greedy decode, compared against the **FP16 output on the
  same card and framework** as reference.
- Report an explicit number — exact-match rate on greedy continuations, or
  perplexity on a held-out passage. Not a vibe check.
- A precision that fails is reported as **rejected on quality**, with the
  number, and excluded from the throughput figure rather than quietly omitted.

Decide the threshold **before** running the ladder. A threshold chosen after
seeing the numbers is not a threshold.

### 4.5 Cost normalisation needs a price recorded on the day

Throughput per GPU-dollar-hour is the headline axis, so the denominator is part
of the result.

- Record the **exact hourly price, the vendor, and the date** for each card, in
  `LOG.md` and in the run record. Prices move; a figure with an unrecorded
  denominator is unreproducible.
- Use one vendor's on-demand pricing for both cards. Mixing a spot A100 with an
  on-demand L4 produces a ratio that means nothing.
- Report raw throughput **alongside** the normalised figure. A reader who
  disagrees with the price can redo the division; a reader given only the ratio
  cannot.

---

## 5. Sizing — what fits in a session

The block runs on Colab Pro, same as M01 and M02, with the same session-length
constraint that shaped both.

| Item | Estimate | Basis |
|---|---|---|
| vLLM cold start + model load | 2–5 min | Weight load dominates |
| One curve point (3 repeats, ~60s window each) | ~5 min | Including warmup and settle |
| One full 8-point concurrency curve | ~40 min | The unit of work to protect |
| A full framework × card sweep at FP16 | ~40 min | One curve |
| **TensorRT-LLM engine build** | **30 min – several hours** | The unknown. §6 |

**The engine build is the only item that can exceed a session.** Everything else
is curve-sized, and a curve fits comfortably.

**Therefore: engines are built once and persisted to Drive, never rebuilt.** An
engine lost to a disconnect is a day lost. Build → immediately copy to Drive →
verify the copy loads → only then sweep.

Confirm before Day 20 that Colab Pro still offers A100 and L4 selection, and
that the L4 has enough disk for weights plus an engine plus a checkpoint copy.

---

## 6. The engine build risk, and how to bound it

This is the M02 labelling problem in a different costume: a step that stands
between the plan and the first number, whose duration is not known in advance.
M02's mistake was discovering the size of that step late. Do not repeat it.

**Day 18 carries a 30-minute spike, not a full build.** Before writing any
harness code, attempt to install TensorRT-LLM and build an engine for the
*smallest available* model — not the study model. The question that spike
answers is not "does my model build" but "**does this toolchain function on this
runtime at all**". If the install itself fails, that is known on Day 18 with
seven days left, not on Day 20 with five.

**Hard stop on Day 21.** If both engines are not serving by end of Day 21, the
TensorRT-LLM axis is dropped and the fallback in §17 is executed. This deadline
is written into `SCOPE.md` on Day 18, before it is inconvenient.

**Engine build time is a reported metric, not overhead.** A framework that is
faster at serving but costs three hours per card per precision has a real
operational cost, and it is one nobody publishes. Record every build: wall time,
peak host RAM, peak VRAM, and whether it succeeded. A table of build times and
failures is a genuine contribution on its own.

---

## 7. What Measurements 01 and 02 already paid for

Carried forward, not rebuilt:

| Asset | From | Use here |
|---|---|---|
| `pins.py` — model revision resolution | M01, extended in M02 | Pin the served model's exact revision. One model, two frameworks: the revision *must* match or the comparison is void |
| Provenance stamping — git SHA, GPU, driver, CUDA, library versions per record | M01 | Same fields, plus framework version and engine build ID |
| One record per **repeat**, aggregated at read time | M02 | Keeps run-to-run spread recoverable. Needed here: a crossover inside the noise band is a *partial* result, and that can only be shown with per-repeat data |
| p50/p95, never mean | M01 | TTFT and TPOT distributions are right-skewed. A mean TTFT is a misleading number |
| `SCOPE.md` written before data, allowed to supersede the plan | M02 | The narrowing discipline. Use it early if needed, not on Day 24 |
| Pareto/dominance analysis and plotting code | M01 | Reusable for the cost-normalised frontier |
| A provenance check must call the same function that wrote the field | M01, the false-alarm day | Do not write a second hash function |

**And the M02 lesson specifically:** the constraint that kills a block is the one
between the plan and the first number. In M02 it was labelling. Here it is the
engine build, and §6 exists to bound it on Day 18 rather than discover it on
Day 20.

---

## 8. Metrics

| Metric | Family | Definition |
|---|---|---|
| **Output tokens/sec** | Throughput | Total output tokens ÷ measurement window, across all concurrent requests. **Headline numerator** |
| **Tokens per dollar-hour** | Cost | Output tokens/sec ÷ host hourly price. **Headline y-axis** |
| **TTFT p50 / p95** | Latency | Client-side, request sent → first streamed chunk. Dominated by prefill and queueing |
| **TPOT p50 / p95** | Latency | Client-side, (last − first chunk) ÷ (output tokens − 1). Dominated by decode bandwidth |
| **End-to-end latency p50 / p95** | Latency | What a user experiences |
| **Goodput** | Throughput | Requests/sec completing under an SLO (e.g. TTFT < 1s **and** TPOT < 50ms). The number an operator actually buys |
| **Peak VRAM** | Cost | High-water mark. Explains where a card runs out of concurrency |
| **KV cache utilisation** | Diagnostic | The counter-space number from the analogy. Explains *why* a curve flattens |
| **Engine build time** | Cost | Wall seconds per engine. TensorRT-LLM only; vLLM records 0 with a note |
| **Quality check** | Guard | Per precision. Pass/fail plus the number behind it (§4.4) |
| **Client CPU utilisation** | Validity | Guards against measuring the harness (§4.2) |

### Record schema — one JSON line per **repeat**

```json
{
  "config":    {"framework":"vllm","framework_version":"0.x.y","card":"A100-SXM4-40GB",
                "concurrency":32,"precision":"fp16","model":"...","max_model_len":4096,
                "input_tokens":512,"output_tokens":128,"greedy":true},
  "config_id": "sha1:...",
  "throughput":{"output_tok_s":2140.5,"goodput_req_s":11.8,
                "slo":{"ttft_ms":1000,"tpot_ms":50}},
  "latency":   {"ttft_ms":{"p50":412,"p95":903},
                "tpot_ms":{"p50":18.4,"p95":31.2},
                "e2e_ms":{"p50":2760,"p95":4410}},
  "cost":      {"usd_per_hour":3.67,"price_source":"...","price_date":"2026-08-23",
                "tok_per_usd_hour":583000},
  "resources": {"peak_vram_mb":36210,"kv_cache_util_pct":88.2,
                "gpu_util_pct":94.1,"client_cpu_pct":41.0},
  "build":     {"engine_build_s":0,"engine_id":null},
  "quality":   {"checked":true,"reference":"fp16","exact_match":1.00,"passed":true},
  "run":       {"n_requests":384,"warmup_requests":32,"window_s":60,
                "repeat":2,"of_repeats":3,"seed":1337,"status":"ok"},
  "prov":      {"gpu":"...","driver":"...","cuda":"...","torch":"...",
                "git_sha":"...","model_revision":"...","prompts_sha":"..."}
}
```

`prompts_sha` is M03's `corpus_sha` — the fixed prompt set, hashed, so a run can
be tied to the exact inputs that produced it.

---

## 9. Day 18 — Tue 18 Aug · Scope freeze, vLLM serving, and the toolchain spike

Two jobs today and they are both gates. `SCOPE.md` fixes what is being claimed
before any number exists. The TensorRT-LLM spike finds out whether the risky
half of the study is possible while there is still time to react.

### Build

New: `SCOPE.md` · `serve_vllm.py` · `client.py` (skeleton) · `prompts.py` ·
`quality.py` — existing: `pins.py`, provenance helpers from M01/M02

- **`SCOPE.md` committed first**, before any code. In scope, out of scope,
  Hypothesis 3 verbatim with its refutation condition, **the chosen direction
  (§1.1)**, and the **Day 21 engine hard stop** (§6).
- **Model chosen and pinned** through `pins.py`. Record the revision. Do the L4
  memory arithmetic explicitly — weights + KV cache at concurrency 128 must fit
  in 24 GB, and if it does not, that constrains the model *now*, not on Day 23.
- **`prompts.py`** — fixed prompt set, fixed input length, hashed to
  `prompts_sha`. Frozen today and never edited again.
- **vLLM serving on A100** with an OpenAI-compatible endpoint, health check, and
  explicit warmup.
- **Single-request sanity number:** tokens/sec at concurrency 1, checked against
  a published figure for the same model class. Order-of-magnitude agreement is
  enough. Disagreement by 10× means something is wrong today, not on Day 23.
- **TensorRT-LLM spike, timeboxed to 30 minutes** (§6). Smallest available
  model, not the study model. The question is whether the toolchain installs and
  builds *anything* on this runtime.

### Write

- **§2 Related work — in full.** The argument is drafted in `REFERENCE.md` §4.3
  and it is a positioning argument, so it goes in early. Paragraph one:
  kernel-level quantization work (APEX4, AnyBCQ, QServe) — establish that
  hardware-dependent efficiency is real and that **this study is deliberately
  not competing there**. Paragraph two: framework-level comparisons are vendor
  blog posts at single operating points, rarely cost-normalised. Close on the
  gap sentence: *whether the framework ranking inverts with concurrency, and
  whether that inversion point moves with hardware tier, is unmeasured.*
- **§1 Introduction — hypothesis paragraph only.**
- **§3.1 System under test.** Model and revision, both cards with their
  architecture generation, framework versions, prompt set stats, `prompts_sha`.
- **§3.2 Sweep table** with the held-fixed controls listed as controls (§3).
- **`LOG.md`** — the concurrency-1 number, the spike outcome with its exact
  error if it failed, and the chosen hypothesis direction with the reasoning.

> **The number:** tokens/sec at concurrency 1 on vLLM/A100, plus a yes/no on
> whether TensorRT-LLM installs at all.

---

## 10. Day 19 — Wed 19 Aug · The load harness, and proving it is not the bottleneck

The harness is the instrument. Today is spent making sure it measures the server
rather than itself. Nothing measured today with an unvalidated client counts.

### Build

New: `client.py` (complete) · `mock_server.py` · `sweep.py` · `record.py`

- **`client.py`** — `asyncio` + `aiohttp`, closed-loop, N workers each holding
  one request in flight (§4.2). Streaming, with per-chunk arrival timestamps.
- **TTFT / TPOT computed exactly as §4.1 specifies**, including the `- 1`.
- **`mock_server.py`** — returns a canned token stream on a fixed sleep. **The
  validity gate:** sweep the mock at every concurrency level and confirm
  measured throughput tracks its theoretical ceiling. If it does not at 128, the
  client is the bottleneck and the top of the axis is not measurable yet.
- **Client CPU utilisation sampled into every record.**
- **`sweep.py`** — drives the concurrency ladder, warmup then fixed window, 3
  repeats, writes one record per repeat (§8).
- **First real curve: vLLM on A100, FP16, all 8 concurrency levels.**

### Write

- **§3.3 Measurement protocol — the section reviewers attack.** Closed-loop
  design and why, warmup policy, window length, client-side latency definitions
  with the formulas, p50/p95 and why not means, repeat count.
- **§3.4 Harness validation.** The mock-server result, stated as a number. This
  paragraph is short and it pre-empts the first serious objection to the study.
- **`LOG.md`** — mock ceiling vs achieved, client CPU at 128, anything the first
  real curve did that was surprising.

> **The number:** one clean vLLM/A100 concurrency curve, and a harness proven to
> hold 128 streams.

---

## 11. Day 20 — Thu 20 Aug · TensorRT-LLM engine, A100 · budget the whole day

`REFERENCE.md` §7 allocates this entire day to one artifact. Do not schedule
anything else against it, and do not treat spare time as a reason to start Day
21 early — a second build attempt with better notes is worth more.

### Build and run

- **Engine build for the study model, A100, FP16.** Log every step: the exact
  command, the container or wheel, versions, wall time, peak host RAM, peak
  VRAM.
- **Persist the engine to Drive immediately on success**, then verify the
  *copy* loads (§5). An engine that exists only in a runtime does not exist.
- **Serve it behind the same OpenAI-compatible endpoint** the vLLM run used, so
  `client.py` drives it unchanged (§4.1).
- **Parity check before any sweep:** same prompt, greedy, TensorRT-LLM output vs
  vLLM output. Divergence beyond sampling noise means the engine is not serving
  the same model — stop and fix, because every number after this depends on it.
- If time remains: the TensorRT-LLM/A100 FP16 concurrency curve.

### Write

- **§3.5 Engine build procedure**, in enough detail to reproduce — this is the
  part vendor blog posts omit, and it is a real contribution.
- **§4 Results — the build-time table**, started today. Card, precision, wall
  time, outcome.
- **`LOG.md`** — every failed attempt with its error, not just the successful
  one. The failure list is what makes the build-time table credible.

> **The number:** an engine that serves, a verified Drive copy, and a build time.

---

## 12. Day 21 — Fri 21 Aug · TensorRT-LLM engine, L4 · fresh build, and the hard stop

**A fresh build, not a copy.** TensorRT-LLM engines are architecture-specific;
an A100 engine will not run on an L4. That constraint is itself a finding worth
one sentence in the paper — a deployment artifact that does not port across
tiers is exactly the hardware-dependence the thesis is about.

**This is the §6 hard stop.** If the L4 engine is not serving by end of day,
execute the §17 fallback tomorrow morning. Do not spend Day 22 on it.

### Build and run

- **L4 engine build.** Same logging as Day 20. Expect it to be slower and
  tighter on memory — the L4 has 24 GB and less host RAM on most runtimes.
- **Record every refusal with its exact error** (`REFERENCE.md` §7). If the
  build fails for a stated reason — insufficient memory, unsupported plugin,
  architecture mismatch — that error string goes in the paper.
- **Both TensorRT-LLM FP16 concurrency curves** if the engine lands early.
- **vLLM on L4, FP16, full curve** regardless of how the engine goes. This is
  cheap, it is not at risk, and it is the curve the fallback design needs.

### Write

- **§4 build-time table extended** with the L4 row and the portability note.
- **§5 Threats — the engine-build entry.** Build times are runtime-dependent and
  were measured on Colab Pro, not bare metal.
- **`LOG.md`** — the hard-stop decision, explicitly, with the time it was taken.

> **The number:** a second engine and a refusal log — or a fallback decision
> taken on time.

---

## 13. Day 22 — Sat 22 Aug · Precision ladder, both cards

The supporting evidence, and the first thing to cut if the schedule has slipped.
Three concurrency levels only — 1, 16, 128 — not the full ladder.

### Build and run

- **`quality.py` gate runs before any precision's throughput is recorded**
  (§4.4). Fixed prompts, greedy, compared against FP16 on the same card and
  framework. Threshold decided **today, before the runs**.
- **FP8, INT8, INT4 where supported.** Expect the A100 (Ampere) to refuse FP8
  and the L4 (Ada) to accept it (§2.3) — verify rather than assume, and capture
  the error text either way.
- **Restart the server between precision changes** (§4.3).
- **Engine build per precision on TensorRT-LLM**, each one timed into the build
  table.

### Write

- **§4 precision table.** Throughput, quality number, pass/fail, build time, and
  refusals as first-class rows rather than blanks.
- **§4 prose on the precision asymmetry.** If FP8 works on the cheaper card and
  not the expensive one, that sentence is the most quotable in the paper. Write
  it plainly and do not oversell it.
- **`LOG.md`** — every refusal error verbatim.

> **The number:** throughput and a quality verdict per precision, per card,
> refusals included.

---

## 14. Day 23 — Sun 23 Aug · Full concurrency sweep, four combinations

The measurement day. Everything before this was building the instrument.

### Build and run

- **Four complete FP16 curves**, 8 concurrency levels, 3 repeats each: vLLM/A100,
  vLLM/L4, TRT/A100, TRT/L4.
- **Re-run any curve collected before the harness was final.** A curve measured
  with a different client version does not belong on the same axes.
- **Watch for OOM cliffs**, especially L4 at 64 and 128. A curve that stops is a
  result — record the concurrency at which each configuration fails and why. An
  OOM ceiling *is* the counter-space limit from the analogy, and it is more
  interpretable than a smooth curve.
- **KV cache utilisation recorded at every point.** It is the diagnostic that
  explains a flattening curve, and without it the paper can describe the shape
  but not the mechanism.

### Write

- **§4 the headline figure**, first draft. Four curves, log x, cost-normalised y.
- **§4 the raw-throughput companion figure** (§4.5).
- **`LOG.md`** — every point that failed, with the concurrency and the error.

> **The number:** four complete curves with per-repeat spread.

---

## 15. Day 24 — Mon 24 Aug · Cost normalisation, crossovers, verdict

### Build and run

- **`analysis.py`** — aggregate per-repeat records, compute p50/p95, apply the
  cost model, locate crossings.
- **Record the prices** — vendor, hourly rate, date, one vendor for both cards
  (§4.5).
- **Locate each crossover and put an interval on it.** A crossover is where two
  curves cross, and with 3 repeats there is a spread. State it as *between 16
  and 32* rather than a false-precision point value.
- **The verdict, against the pre-registered condition and nothing else:**
  - Different crossover concurrency per card, outside the spread → **confirmed**
  - Same crossover on both, or one framework leading everywhere on both →
    **refuted**
  - Crossover moves but within the spread → **partial**, and say so
- **Sensitivity check, carried from M01.** Vary the assumed price ±30% and
  report whether the ordering survives. The M01 reviewers asked for exactly this
  kind of robustness and it cost almost nothing.

### Write

- **§4 Results, complete**, with the verdict stated in the pre-registered
  language.
- **§5 Threats:** single runtime provider; Colab-tier hardware sharing;
  prices at one date from one vendor; one model, one size; closed-loop load only;
  quality checked on a fixed prompt set rather than a benchmark suite.
- **`LOG.md`** — crossover locations and the sensitivity result.

> **The number:** a crossover concurrency per card, with an interval, and a
> verdict.

---

## 16. Day 25 — Tue 25 Aug · Paper 3, writing only

No new measurements. If a number is missing today, it stays missing and is
described as missing.

### Already done — assemble

§1 hypothesis paragraph (Day 18) · §2 Related work (Day 18) · §3.1–3.2 (Day 18) ·
§3.3–3.4 protocol and harness validation (Day 19) · §3.5 engine builds (Day 20) ·
§4 tables and figures (Days 20–24) · §5 Threats (Days 21, 24)

### Write today

- **§1 Introduction, in full.** Open on the practitioner situation: framework
  choice is made once, from a benchmark run at one operating point, and then
  deployed across whatever hardware is available. Then the gap, then the
  hypothesis, then the contribution.
- **§6 Conclusion**, including the honest scope statement: one model, one
  provider, two cards, one load pattern.
- **The programme paragraph.** Paper 3 is the second of two completed hardware
  comparisons, not the third of three. Reference `Ingestion/LOG.md` Day 3 rather
  than passing over it — a study that stopped for a stated reason reads better
  than a study that is silently absent.
- **Verify every citation** against the primary source, as in M01. The METIS
  SOSP '25 venue is still unverified in M01's bibliography — settle it here and
  fix it in both.
- **LaTeX build**, same template as Paper 1.

> **The number:** a finished draft with every citation checked.

---

## 17. Contingency

| If | Then |
|---|---|
| TensorRT-LLM will not install on the runtime (known Day 18) | Switch the framework axis to **vLLM configuration**: KV cache fraction, max-num-seqs, chunked prefill on/off. Still a concurrency sweep, still two cards, still a crossover question. Rewrite `SCOPE.md` on Day 18 — early, cheap, and honest |
| An engine builds on A100 but not L4 (Day 21 hard stop) | Report the A100 crossover, and report the L4 build failure with its error as a **result about portability**. Do not fake a comparison. Add a vLLM-only L4 concurrency curve so the card axis still has data |
| The client cannot hold 128 streams (Day 19) | Cap the axis at the highest validated concurrency and say so in §3.4. A truncated but valid axis beats an invalid point |
| L4 OOMs above concurrency 32 | That is the finding. Record the ceiling and the KV cache number behind it. A hard capacity limit is a stronger hardware-dependence result than a gentle curve |
| A precision passes throughput but fails quality | Report it as **rejected on quality** with the number. Never quietly drop it |
| Throughput differs but the curves never cross | Refuted, cleanly. Report it. "The ranking is stable across concurrency on both cards" is a useful negative for anyone choosing a framework |
| Colab will not allocate an L4 (or an A100) | Try the alternate tier available and rename the axis honestly. Two tiers that differ is what the hypothesis needs; the specific model numbers are not sacred |
| Slipping more than two days | Cut the precision ladder entirely, then the 64/128 rungs. Protect: both cards, both frameworks, FP16, concurrency 1–32 |
| Day 25 arrives with the sweep incomplete | Write on what exists and state the reduced design plainly. There is no Block 4 to absorb the slip — the programme ends here |

---

## 18. Open decisions

Settle these before 18 August.

- **Hypothesis direction (§1.1).** vLLM-low/TRT-high as `REFERENCE.md` §7
  commits, or the reverse. **This is the highest-value decision in the block**
  and it must be made before Day 18 ends. Write the reasoning in `LOG.md`, not
  just the choice.
- **Model — the M03 equivalent of M02's embedder choice, and as consequential.**
  It must fit on the L4 at concurrency 128 with a usable KV cache, be supported
  by *both* frameworks, and be quantizable to the precisions in the ladder. Do
  the 24 GB arithmetic before committing. A 7–8B instruct model is the working
  assumption; a 3B fallback removes memory pressure at the cost of making the
  L4's ceiling less interesting.
- **Input and output token lengths.** They set prefill/decode balance and
  therefore where the crossover lands. Pick one representative pair, fix it, and
  state it as a scope limit in §5 rather than sweeping it.
- **SLO thresholds for goodput.** TTFT < 1s and TPOT < 50ms are the working
  assumption. Decide before Day 23; a threshold chosen after seeing curves is
  not a threshold.
- **Quality gate threshold** (§4.4). Metric and pass mark, decided Day 22 before
  the ladder runs.
- **Pricing source.** One vendor, on-demand, both cards, recorded with the date.
- **Measurement window length and warmup count.** 60 s and 32 requests are the
  working assumptions; confirm against the Day 19 curve.

---

*Companion to `research/m03-serving-plan.html`. Policy lives in `REFERENCE.md`
— amend that file, not this one. Ground truth here is machine-generated; there
is no annotation pass, which is the stated reason this block follows the
Measurement 02 halt (`Ingestion/LOG.md`, Day 3).*
