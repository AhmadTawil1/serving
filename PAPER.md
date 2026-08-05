# Where does the vLLM / TensorRT-LLM crossover sit, and does it move with the GPU?

Draft. Sections are filled in on the day the thing they describe is built or
measured (`M03-SERVING.md`, day-by-day plan) — a section with no content yet
is marked `[pending: day N]`.

---

## Abstract

[pending: day 7 (block-relative — "day 25" in the programme's original
numbering)]

---

## 1. Introduction

Choosing a serving framework is usually a one-time decision made from a
vendor benchmark: one model, one batch size, one operating point, on
whatever GPU the benchmark happened to run on. The choice is then deployed
wherever budget allows, which is rarely the card the benchmark used. Two
questions follow from that gap, and neither is answered by a single
operating point: does the ranking between frameworks hold as concurrent load
increases, and does wherever it stops holding move when the hardware does?

**Hypothesis 3, registered before any data was collected.**

> vLLM leads at low concurrency; TensorRT-LLM leads above a crossover point;
> and **the crossover moves with GPU tier**, arriving at a different
> concurrency on the L4 than on the A100, because compiled engine-level
> optimisation and memory-efficient batching are rewarded differently by a
> card with less bandwidth and less headroom.

**Direction fixed before any sweep** (`SCOPE.md`): vLLM-low / TensorRT-LLM
-high. At concurrency 1 there is nothing to batch and decode is
bandwidth-bound, so neither framework can compile its way past the memory
wall — they should be close. TensorRT-LLM's fused kernels and in-flight
batching are load-time optimisations, and they should show up under load,
not at concurrency 1.

**Refuted if.** One framework leads at every concurrency level on both
cards, or the crossover sits at the same concurrency on both.

**Partial if.** A crossover exists and moves, but the movement is within
run-to-run variance, or the mechanism behind it is not the one predicted
(e.g. an OOM cliff rather than a gradual compute/bandwidth trade).

[pending: block day 7 — contributions as three bullets, and the outcome
sentence once the verdict exists]

---

## 2. Related work

Serving-framework comparison sits directly beneath a body of work that is
well-resourced and moving fast, and the positioning has to say precisely
where this study is not competing.

**Kernel-level quantization work is active and not the target here.** APEX4
(Jun 2026) shows that the tensor-core-to-CUDA-core ratio is the primary
hardware factor governing W4A4 kernel efficiency — the same kernel gives
2.0–2.5× on an RTX 3090 but only 0.43–0.47× on an A100, so viability is
platform-dependent rather than universally infeasible [CITE]. AnyBCQ shows
relative quantization speedups are preserved across A100 and H100 [CITE].
QServe co-designs the algorithm and system layers for quantized serving
[CITE]. APEX4 in particular is, in effect, the kernel-level version of this
entire thesis, published two months before this measurement — welcome
confirmation that hardware-dependent efficiency is real, and a fence: the
kernel angle is taken, and this study does not attempt it again one layer
down.

**What remains thin is serving-*framework* choice under concurrency.**
Public vLLM-versus-TensorRT-LLM comparisons are vendor blog posts reporting
single operating points — one batch size, one prompt length, rarely
cost-normalised against the hardware each result ran on. None sweeps
concurrency as an axis, and none asks whether the ranking they report holds
on a different card. Multi-adapter LoRA serving overhead is thinner still
and out of scope here.

> **Gap.** Framework-level serving comparisons report single operating
> points; whether the framework ranking inverts with concurrency, and
> whether that inversion point moves with hardware tier, is unmeasured.

This report measures the crossover directly, cost-normalised, on two cards
five bandwidth-generations apart.

---

## 3. Method

### 3.1 System under test

**Model.** `Qwen/Qwen2.5-7B-Instruct`, revision `a09a35458c702b33eeacc393d103063234e8bc28`
(`configs/model_pins.yaml`, resolved via `serving/pins.py`). Chosen over an
8B Llama-class model for two reasons. First, grouped-query attention with
only 4 KV heads (vs. Llama-3.1-8B's 8) roughly halves the per-token KV-cache
footprint, which is what makes concurrency 128 fit on a 24 GB L4 at all —
see the worked arithmetic in `serving/config.py`: weights (~14.2 GiB, bf16)
plus KV cache at concurrency 128 and 640 tokens/request (~4.4 GiB) leaves
~5.4 GiB of the L4's 24 GiB for activations and framework overhead, tight
but workable, against ~21.4 GiB of headroom on the 40 GiB A100. Second, the
model is ungated on the Hub — a gated alternative (Llama-3.1-8B-Instruct,
confirmed gated by a live 401 while checking its config) would tie
reproduction to one account's license acceptance, an unnecessary
reproducibility dependency for a study that is not about the model at all.

**Cards.** A100-SXM4-40GB (Ampere, SM80, ~1,555 GB/s) and L4-24GB (Ada,
SM89, ~300 GB/s) — the same pair used in Measurement 01, rented via Colab
Pro (`REFERENCE.md` §3). Bandwidth ratio ~5.2×.

**Frameworks.** vLLM and TensorRT-LLM, both serving the pinned revision
behind an OpenAI-compatible completions endpoint, driven by the same client
(`scripts/client.py`) so no framework-specific request code can become a
confound (§4.1). Both are installed in their own environment on the GPU
host, not in this project's own `uv` environment — see
`configs/serve-env.md` for why (version pins that conflict with this
project's own tooling, and neither publishes a Windows wheel — confirmed 5
Aug 2026 by a direct install attempt, recorded in `LOG.md`).

**Prompt set.** 8 fixed prompts (`data/prompts.json`, `serving/prompts.py`),
topically varied (systems explanation, code, arithmetic reasoning, incident
writing, product copy) so the Day 22 quality gate has real content to
compare, each padded with a deterministic filler passage to exactly 512
input tokens under the study model's own tokenizer. `prompts_sha` (SHA1 over
the frozen file) is stamped into every record's `prov` block. Output length
is fixed at 128 tokens and forced with `min_tokens` so no framework can look
faster by stopping early (§4.1) — see the record schema in §3.2.

### 3.2 Sweep

| Axis | Levels |
|---|---|
| Framework | vLLM · TensorRT-LLM |
| Card | A100-SXM4-40GB · L4-24GB |
| Concurrency | 1 · 2 · 4 · 8 · 16 · 32 · 64 · 128 (log-spaced) |
| Precision | FP16 · FP8 · INT8 · INT4, where the architecture permits |

Main sweep: 2 frameworks × 2 cards × 8 concurrency levels × 3 repeats, all
FP16 = 96 measured runs. Precision ladder: 4 precisions × 2 frameworks × 2
cards × 3 concurrency levels (1, 16, 128) × 3 repeats = 144 runs minus
whatever each card refuses — refusals are recorded with their exact error,
not treated as missing cells (`REFERENCE.md` §7).

**Held fixed, and reported as configuration:**

| Held | Value | Why it matters |
|---|---|---|
| Model | `Qwen/Qwen2.5-7B-Instruct`, one pinned revision | Two frameworks serving different revisions would not be a comparison |
| Input length | 512 tokens, fixed prompt set | Prefill cost scales with it |
| Output length | 128 tokens, forced via `min_tokens` | A framework stopping early would look faster for free |
| Sampling | Greedy (`temperature=0`) | Removes a variance source; makes the quality gate deterministic |
| Max model length | 2048, identical on both frameworks | Changes achievable KV-cache concurrency if it moves |
| Request pattern | Closed-loop, N workers, each 1 request in flight | §4.2 — this is what "concurrency N" means here |
| Warmup | Fixed request count at the target concurrency, discarded | §4.3 — first requests are engine artifacts (CUDA graph capture, lazy kernel compilation) |

**Record schema.** One JSON line per repeat; full schema and field
definitions in `M03-SERVING.md` §8. `prov` is stamped by `serving/provenance.py`
and carries `gpu`, `driver`, `cuda`, `torch`, `git_sha`, `model_revision`,
`prompts_sha` — framework and framework_version live in the record's
`config` block instead, since they vary per run rather than per environment.

[pending: block day 2 (harness/metrics/provenance detail once `sweep.py` and
`record.py` exist) — §3.3 measurement protocol, §3.4 harness validation]

---

## 4. Results

[pending: block days 5–6 (labelled "Day 23–24" in the programme's original
numbering)]

---

## 5. Threats to validity

[pending: block day 7]

- Rented hosts have noisy neighbours; variance across the 3 repeats will be
  reported per §7 of `REFERENCE.md`, not smoothed over.
- One model, one size. `Qwen/Qwen2.5-7B-Instruct` was chosen for its L4
  memory arithmetic and its ungated distribution, not because it is
  representative of every model class a practitioner might serve.
- Concurrency sweep is synthetic closed-loop load, not production traffic
  (§4.2 explains the design choice; the honest limit is that no real
  arrival-time distribution is modelled).
- Single seed, greedy decoding throughout — removes a variance source at the
  cost of not characterising sampling-mode latency.
- Colab-tier hardware sharing and Colab-provisioned SKUs (§3 of
  `REFERENCE.md` — the A100 is the 40GB SXM4 SKU, not the 80GB variant).

---

## 6. Conclusion

[pending: block day 7]

---

## References

[pending — to verify against primary sources before the block's writing day,
per `REFERENCE.md` §15: APEX4, AnyBCQ, QServe, the TensorRT-LLM support
matrix, the vLLM PagedAttention paper]
