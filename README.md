# Does the best LLM serving configuration depend on the GPU it runs on?

A controlled measurement of whether a vLLM tuning decision made on one GPU
transfers to another, and whether the answer changes with load.

<p align="left">
  <a href="paper/main.pdf">
    <img src="https://img.shields.io/badge/Paper-PDF-red?style=for-the-badge&logo=adobeacrobatreader&logoColor=white" alt="Read the paper (PDF)">
  </a>
</p>

---

## The question

Serving an LLM means choosing a configuration — KV-cache budget, scheduler
limits, whether to reuse cached computation across requests sharing a prefix.
That choice is usually made once, on whatever GPU the team has, and then carried
onto whatever hardware turns out to be affordable.

**Is that carry-over safe?** If not, every published tuning recommendation has
an unstated hardware precondition attached to it.

## The answer

Two findings, pointing in opposite directions.

**1. The configuration payoff transfers.** Prefix caching is worth ~1% at
concurrency 1 and rises monotonically to 2.56× on an A100 and 2.70× on an L4 —
two cards that differ by 4.35–5.26× in raw throughput. The curves track each
other at every level in between.

| conc | 1 | 2 | 4 | 8 | 16 | 32 | 64 | 128 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **A100** | 1.01× | 1.03× | 1.07× | 1.14× | 1.28× | 1.53× | 1.94× | **2.56×** |
| **L4** | 1.01× | 1.03× | 1.06× | 1.13× | 1.31× | 1.39× | 1.84× | **2.70×** |

*Throughput with prefix caching ÷ without.*

**2. Hardware viability does not transfer.** Under an SLO fixed before any
measurement (TTFT < 1000 ms **and** TPOT < 50 ms), the A100 complies at every
concurrency level and the **L4 complies at none** — under either configuration.
Its per-token latency is 56.6 ms with a single user, already past the limit.

> **The relative recommendation transfers; the absolute viability does not.**
> *How should I configure the server?* can be answered on any card.
> *Can this card serve my workload?* cannot be answered anywhere but on the card.

**The hypothesis under test was falsified.** It predicted that the best
configuration would change with load, and that the change point would move with
the card. Prefix caching leads at every level on both cards — 16 comparisons,
zero counter-examples — so no change point exists to move. The falsification
criterion is applied mechanically by `scripts/analysis.py`, not by judgement.

## Figures

| | |
|---|---|
| [`figs/fig2_payoff.pdf`](figs/fig2_payoff.pdf) | **The headline.** Payoff ratio vs concurrency, per card. Two lines lying almost on top of each other. No price assumption, no derived quantity. |
| [`figs/fig1_throughput.pdf`](figs/fig1_throughput.pdf) | Throughput vs concurrency, all four curves |
| [`figs/fig3_slo.pdf`](figs/fig3_slo.pdf) | Per-token latency against the SLO ceiling, and goodput. Both L4 curves sit flat on zero |
| [`figs/fig4_cost.pdf`](figs/fig4_cost.pdf) | Cost-normalised throughput, with the break-even GPU price marked |

---

## What was measured

| | |
|---|---|
| **Model** | `Qwen/Qwen2.5-7B-Instruct`, revision `a09a3545…`, pinned and stamped into every record |
| **Framework** | vLLM 0.26.0, OpenAI-compatible endpoint |
| **Cards** | A100-SXM4-40GB · L4 24 GB (Colab Pro) |
| **Swept** | prefix caching on/off × card × concurrency {1,2,4,8,16,32,64,128} |
| **Held fixed** | 512 input / 128 output tokens (forced), greedy decoding, max model length 2048, fp16 |
| **Records** | 80, all successful. 3 repeats per point on A100, 2 on L4 |
| **Run-to-run spread** | ≤ 1.66% anywhere in the grid; 0.004% at concurrency 1 |

### Method notes that matter

- **Closed-loop load.** N workers, each holding exactly one request in flight.
  In-flight count is exactly N at all times.
- **Latency measured client-side, on the stream.** `TTFT = t1 − t0`;
  `TPOT = (t2 − t1) / (n − 1)`. The `n − 1` is deliberate: the first token is
  already charged to TTFT, so there are `n − 1` inter-token gaps to average.
- **The load generator was validated before any real number was trusted.** The
  full sweep was first run against a mock server with a known ceiling.
  Throughput tracked linearly with concurrency across the whole range
  (0.93–1.06 of linear) and client CPU peaked at 22.8% under real load — so the
  harness is not the thing being measured. See
  [`results/harness_validation.md`](results/harness_validation.md).
- **One record per repeat**, aggregated at read time, so run-to-run spread stays
  recoverable rather than being averaged away at write time.

---

## Reproducing

Requires Python 3.12 and [`uv`](https://docs.astral.sh/uv/).

### Analysis and figures — no GPU needed

```bash
uv sync
uv run python scripts/analysis.py                  # every number in the paper
uv run python scripts/analysis.py --emit markdown  # paste-ready tables
uv run python scripts/plot_results.py              # regenerate figs/
uv run pytest                                      # 88 tests
```

`analysis.py` reads the four result files in `results/` and prints the tables,
the verdict against the falsification criterion, and the price sensitivity.
**Every number in the
paper is produced by it** — none is typed by hand. It refuses to run if the
frozen inputs (model, revision, prompt hash, token lengths, sampling) are not
identical across all four result files.

Prices are command-line arguments, not constants, because they are not measured:

```bash
uv run python scripts/analysis.py --price-a100 3.67 --price-l4 0.70 --sensitivity 0.30
```

### Collecting new measurements — GPU required

vLLM is deliberately **not** a dependency of this project: it pins
`torch`/`transformers` versions that conflict with the tooling here, and has no
Windows wheel. It is installed separately on the GPU host — see
[`configs/serve-env.md`](configs/serve-env.md).

```bash
pip install vllm
# add --prefix-caching for the ON curve
python scripts/serve_vllm.py --card A100-SXM4-40GB --precision fp16
python scripts/sweep.py --base-url http://localhost:8000 \
    --framework vllm --card A100-SXM4-40GB --precision fp16 \
    --window-s 60 --repeats 3 --warmup-requests 32 \
    --out results/<name>.jsonl
```

### Building the paper

```bash
cd paper
pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
```

The bibliography is embedded via `filecontents`, so Overleaf needs only
`main.tex` and the `figs/` directory.

---

## Repository layout

```
serving/     library code — config, model pinning, provenance, prompt
             freezing, record schema, server lifecycle
scripts/     executables — server launch, load client, sweep driver,
             mock server, analysis, plotting
tests/       88 tests, all runnable without a GPU
configs/     model pin (repo → exact revision), GPU-host setup notes
data/        frozen prompt set + hash; mock-server validity-gate records
results/     the four measured curves (80 records) + harness validation
figs/        generated figures
paper/       LaTeX source, embedded bibliography, compiled PDF
```

### Result record schema

One JSON object per repeat, in `results/*.jsonl`:

```json
{
  "config":    {"framework":"vllm","card":"A100-SXM4-40GB","concurrency":32,
                "precision":"fp16","model":"...","input_tokens":512,
                "output_tokens":128,"greedy":true},
  "throughput":{"output_tok_s":2329.2,"goodput_req_s":18.20,
                "slo":{"ttft_ms":1000,"tpot_ms":50}},
  "latency":   {"ttft_ms":{"p50":"...","p95":"..."},
                "tpot_ms":{"p50":"...","p95":"..."},
                "e2e_ms":{"p50":"...","p95":"..."}},
  "run":       {"n_requests":1120,"warmup_requests":32,"window_s":60,
                "repeat":2,"of_repeats":3,"seed":1337,"status":"ok"},
  "prov":      {"gpu":"...","driver":"...","cuda":"...","torch":"...",
                "git_sha":"...","model_revision":"...","prompts_sha":"..."}
}
```

---

## Limitations

Stated here as well as in the paper, because they bound what the result means.

- **The cost comparison is too close to call.** At GCP list prices the L4
  delivers 6–20% more tokens per dollar, but the break-even A100 price is
  $3.05–$3.68/hr and the ordering does not survive a ±10% price swing. Prices
  were taken from secondary sources, not confirmed against the vendor's own
  page. `fig4_cost.pdf` plots the break-even so a reader can substitute their
  own contract prices.
- **The prompt set is 8 prompts**, so the caching-on curves reflect a 96.9%
  prefix-cache hit rate. They are a *ceiling* under maximal prompt reuse, not a
  typical production workload. A diverse-prompt workload would sit between the
  two curves.
- **The SLO is a working assumption.** The L4's zero-goodput result is a
  statement about TPOT < 50 ms. At a 100 ms budget the L4 becomes viable at
  every concurrency — but only with caching on. That sensitivity is reported in
  the paper.
- **One model, one size, one sequence-length pair, one provider, shared
  hardware.** Absolute throughput figures are indicative; the ratios, which are
  the contribution, are internally consistent because both members of each ratio
  were measured in the same session on the same allocation.
- **A framework comparison was attempted and abandoned.** TensorRT-LLM 1.2.1
  installs but cannot load its own bindings on the target runtime — it requires
  `libcublasLt.so.13`, which is not obtainable from PyPI or NVIDIA's own index.
  The error and the three attempted remedies are reported in the paper (§3.8)
  rather than omitted, because the failure is a real cost of choosing that
  framework. No engine was built on either card, so no build-time data is
  claimed.

---

## Related measurement

A separate experiment on RAG retrieval pipelines, using the same two cards,
found that most of the latency ranking also transfers between them
(Kendall τ = 0.80) — an independent result pointing the same way as this one.

---

## Citation

```bibtex
@techreport{tawil2026serving,
  author      = {Tawil, Ahmad},
  title       = {Does the best {LLM} serving configuration depend on the
                 {GPU} it runs on? An experiment report},
  year        = {2026},
  institution = {Independent},
  note        = {https://github.com/AhmadTawil1/serving}
}
```

Ahmad Tawil · [ahmadtawil.se@gmail.com](mailto:ahmadtawil.se@gmail.com)
