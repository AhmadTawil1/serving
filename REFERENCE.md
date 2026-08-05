# Configuration Transfer Across Hardware Tiers
### Full working reference — three measurements, four papers

**Author:** Ahmad Tawil
**Window:** 31 Jul → 28 Aug 2026 · 28 working days
**Status:** planning complete, Block 0 not started

This document is the single reference for the whole programme. It holds the thesis, the literature position for each study, the exact experimental design, the schedule, and the structure of all four papers. If something isn't in here, it isn't in scope.

---

## 1. The thesis

> **Does the optimal configuration of an AI pipeline depend on the hardware it runs on, and if so, which classes of configuration knob are hardware-dependent and which are invariant?**

Three independent settings, one question:

| # | Setting | The knob space |
|---|---|---|
| 01 | Retrieval | chunking, top-k, reranking, embedding dimension |
| 02 | Video ingestion | decode path, batching, workers, resolution, frame sampling |
| 03 | LLM serving | serving framework, precision, concurrency |

**Why this question is worth asking.** Configuration-optimisation papers publish a recommended pipeline benchmarked on the hardware the authors had. Practitioners then copy those recommendations onto hardware they can afford. Nobody has systematically tested whether the recommendation survives the move. If it doesn't, every published "optimal configuration" carries an unstated hardware precondition.

**Why it is tractable for one person.** It requires no new model, no new dataset, and no training. It requires one careful harness and disciplined sweeps. Recent work of this shape has been produced for roughly 20 A100-hours — about USD 34 on rented capacity. Budget is not the constraint; day count is.

---

## 2. Ground rules

1. **The prediction is written before the sweep.** Each measurement has a hypothesis and an explicit refutation condition committed to git before data collection begins. Editing a hypothesis after seeing data destroys the result.
2. **A refuted hypothesis ships.** The deliverable is the verdict, not the outcome. Refutations are more interesting than confirmations and easier to defend.
3. **One harness, three studies.** Same runner, same provenance stamping, same statistics. This is what makes Paper 4 possible and is itself a methodological contribution.
4. **Report p50 and p95. Never the mean.** Latency distributions on shared rented hosts are not symmetric.
5. **Stamp everything.** Every JSON record carries GPU name, driver version, library versions, git SHA, and seed. A result without provenance is not a result.
6. **When behind, cut cells from the grid — never the second GPU.** The second GPU is the entire thesis.
7. **Six days measuring, one day writing — but write on all seven.** Day 7 stays a writing-only day, and a different mode of work: sustained building historically fails around day six, and the mode switch resets that. What changes is that days 1–6 each end with writing too. See §10.1.
8. **Papers 1–3 stay off arXiv until Paper 4 is decided.** On a personal site, LinkedIn, and GitHub they are unpublished write-ups that can be folded into a larger submission freely. Once on arXiv, a venue may treat Paper 4 as overlapping prior work.
9. **10 August is a date, not a milestone.** Applications go out that day regardless of state.

---

## 3. Hardware

| Tier | Card | Arch | SM | VRAM | Mem bandwidth | Used in |
|---|---|---|---|---|---|---|
| Strong | A100-SXM4-40GB | Ampere | SM80 | 40 GB | ~1,555 GB/s | 01, 02, 03 |
| Weak | L4 24GB | Ada | SM89 | 24 GB | ~300 GB/s | 01, 02, 03 |
| Optional | T4 16GB | Turing | SM75 | 16 GB | ~320 GB/s | 01 only |

**The bandwidth ratio is ~5.2×.** That spread is what makes configuration crossings visible. Two cards from adjacent generations would produce a null result for uninteresting reasons.

**Amendment, 31 Jul 2026 (Day 4).** The strong tier is provisioned via Colab Pro rather than a rented cloud instance — Colab Pro lets the runtime type be selected explicitly (A100/L4/T4 all available on demand), which satisfies the card-identity requirement below just as well as a rental would. The A100 Colab Pro actually provisions is the **40GB SXM4 SKU** (~1,555 GB/s), not the 80GB SKU (~2,039 GB/s) originally assumed when this table was planned — confirmed via `torch.cuda.get_device_name(0)` and `nvidia-smi`, not assumed. The bandwidth ratio against the L4 is therefore ~5.2×, not ~6.8×; still wide enough for the study's purpose, and the number now on record is the one measured, not planned.

### Critical correction to the original plan

The original 35-day plan paired **A100 with T4** throughout. This does not work for Measurement 03.

TensorRT-LLM's support matrix covers Hopper, Ada and Ampere. Specifically:

- INT4 AWQ and GPTQ require **SM80 or above**
- INT8 SmoothQuant is **unsupported on SM70 and SM75**
- Fused multi-head attention **rejects sm70 and sm75**

The T4 is SM75. Attempting Measurement 03 on it produces two days of toolchain errors and no data. The L4 is the correct substitute: Ada architecture, FP8 available, cheap on RunPod and Vast, and its bandwidth gap against the A100 is larger than the T4's.

Keep the T4 available for Measurement 01 only, where a third weak tier adds a data point and nothing needs TensorRT-LLM.

---

## 4. Literature position

Each study needs a gap sentence. These are the gaps, and the work that surrounds them.

### 4.1 Retrieval — where the field already is

Multi-objective RAG configuration search is **crowded**. Do not propose finding a Pareto frontier; that is done.

| Work | What it does |
|---|---|
| **syftr** | Multi-objective search across agentic and non-agentic RAG configurations, balancing latency, accuracy and cost |
| **RAG-Stack** | Identifies the quality–performance Pareto frontier by navigating the joint algorithm–system design space |
| **RAGSmith** | Evolutionary search over complete pipelines rather than greedy per-module selection; reports dataset sensitivity |
| **METIS** | Joint algorithm/system exploration for retrieval-augmented pipelines |
| Cost–latency–quality benchmarks | A growing line of work treating the three-way tradeoff as the object of study |

**The gap.** All of these produce a recommended configuration on their own hardware. RAG-Stack acknowledges that hardware options exist but explores primarily from the algorithm side. None asks whether the recommendation transfers downward.

**Gap sentence for Paper 1:** *Existing work locates the configuration frontier on fixed hardware; whether the frontier's composition is stable across hardware tiers has not been tested.*

### 4.2 Video ingestion — two literatures that never meet

This is the strongest positioning of the three, because the gap is structural rather than incremental.

**Systems side** — reports throughput, validates with a detection proxy:

- **CoVA** — pushes analysis into the compressed domain, ~4.8× average throughput by unclogging the decode bottleneck
- **2025 industrial case study** — uses OpenCV's `grab`/`retrieve` split so the frame index advances without decoding; sampling 2 FPS from 30 FPS video otherwise decodes and discards 14 of every 15 frames. Validates via average tracklet length and detection coverage
- **ViCoStream** and related streaming pipelines

**Retrieval side** — reports recall, against an abstract frame budget:

- Empirical comparison of frame sampling methods for video RAG — trades sampled-frame count against retrieval recall
- **Shot-aware sampling**, **ProCLIP** and related adaptive-sampling work

**Industry advice with no published number:** Coactive recommends ingesting at 360p/30fps to balance processing speed, storage efficiency and AI performance. No recall figure accompanies this. **Verified 3 Aug 2026** against the primary source (Coactive, *Unlocking Video Insights: How an Optimized Video Ingestion Pipeline Transforms Content Discovery*, https://www.coactive.ai/blog/optimized-video-ingestion-pipeline) rather than taken from a to-locate list.

**The justification is a threshold argument, and it has a hole in it.** The same post explains *why* 360p suffices: analysis models operate at a fixed input resolution and resize every source frame to it before analysis. That is a valid argument — but the post also puts the fixed input size at **anywhere between 224×224 and 1024×1024 depending on the model**. A 4.5× spread. If the model runs at 224, 360p sits safely above the threshold and the advice holds; if it runs at 384 or 1024, 360p sits below it and the pipeline is feeding upscaled mush. The recommendation is stated unconditionally and never says which regime the reader is in.

This sharpens the gap from *folklore versus measurement* to *conditionally-true advice whose condition is never stated, and whose boundary nobody has measured*. It is also what motivates the amended threshold form of Hypothesis 2 (§6).

**The gap.** Systems papers measure accuracy with detection proxies, not semantic retrieval. Retrieval papers measure recall against frame *count*, never against achieved wall-clock throughput on identified hardware. The two knob families have never been placed in one measurement space.

**Gap sentence for Paper 2:** *Throughput optimisations and sampling strategies are evaluated in separate literatures with incompatible cost axes; their recall costs have not been compared on a common frontier.*

### 4.3 LLM serving — heavily occupied at kernel level

Well-resourced labs are active here. **Do not compete on kernel-level quantization.**

| Work | What it establishes |
|---|---|
| **APEX4** (arXiv:2606.08761, Guo et al., Jun 2026) — *verified 5 Aug 2026* | Tensor-core-to-CUDA-core ratio is the primary hardware factor governing W4A4 efficiency. Same kernel gives 2.0–2.5× on RTX 3090 but 0.43–0.47× on A100 — viability is *platform-dependent*, not universally infeasible |
| **AnyBCQ** | ~~Relative quantization speedups preserved across A100 and H100~~ **CORRECTED 5 Aug 2026 — this claim is not made by the paper.** AnyBCQ (arXiv:2510.10467, Park et al., ICLR 2026) is a multi-precision extension of binary-coded quantization with direct bit-plane operations; it reports throughput gains up to 3.0x over half precision and 1.2x over prior multi-precision methods. The A100/H100 attribution was wrong and had propagated into a draft of Paper 3 before being caught |
| **QServe** (arXiv:2405.04532, Lin et al., MLSys 2025) — *verified 5 Aug 2026* | W4A8KV4 quantization and system co-design; reports 2.4-3.5x higher throughput than TensorRT-LLM on A100 and L40S. Note the exact title is "W4A8KV4 Quantization and System Co-design", not "Algorithm and System Co-design" |

APEX4 is, in effect, the kernel-level version of this entire thesis, published two months ago. That is good news — it establishes the direction is real — and a constraint: the kernel angle is taken.

**What remains thin.** Serving-*framework* choice under concurrency. The public vLLM-versus-TensorRT-LLM comparisons are vendor blog posts reporting single operating points, not concurrency sweeps, and rarely cost-normalised. Multi-adapter LoRA serving overhead is thinner still.

**Gap sentence for Paper 3:** *Framework-level serving comparisons report single operating points; whether the framework ranking inverts with concurrency, and whether that inversion point moves with hardware tier, is unmeasured.*

---

## 5. Measurement 01 — Retrieval

**Dates:** Mon 3 Aug → Sun 9 Aug (6 measure + 1 write)

### Question
Does the Pareto-optimal RAG configuration transfer across GPU tiers, or does the ranking of configurations invert?

### Hypothesis 1
The Pareto-optimal RAG configuration is **not hardware-invariant**. Chunk size and top-k retain their optimal settings across tiers. Cross-encoder reranking flips from Pareto-optimal on the A100 to Pareto-dominated on the L4, because its latency cost grows faster than the recall it buys.

### Refuted if
Both frontiers contain the same configurations in the same order. Identical composition means the recommendation transfers and the thesis fails in this setting.

### Configuration grid

| Knob | Levels |
|---|---|
| Chunk size | 256 / 512 / 1024 tokens |
| Chunk overlap | 0 / 15% |
| Top-k | 3 / 5 / 10 / 20 |
| Reranker | off / cross-encoder |
| Embedding dimension | small / base model pair |

Report the exact cell count in the paper. Cut levels, never cards, if time runs short.

### Corpus and ground truth
15 topic PDFs. **30 questions, hand-labelled with the chunk IDs that actually answer them.** Single annotator; state this in threats to validity. Every number in the paper inherits the validity of these labels — do not generate them.

### Metrics
recall@5, recall@10, MRR, context precision · p50 and p95 end-to-end latency · per-stage latency (embed / search / rerank / generate) · peak VRAM · tokens per second.

### Day plan

| Day | Date | Work | The number |
|---|---|---|---|
| 4 | Mon 3 Aug | `SCOPE.md` committed first. Then baseline: recursive splitting, sentence-transformers, vector store with source metadata, answer-only-from-context prompt | Answers in-corpus, refuses out-of-corpus |
| 5 | Tue 4 Aug | The gold set: 30 hand-labelled questions. `eval.py` computing recall@k, MRR, context precision | `eval.py` scores any config |
| 6 | Wed 5 Aug | Instrument the stages: per-stage timing, p50/p95, peak VRAM. One run emits quality *and* cost records | Single run produces both, no manual bookkeeping |
| 7 | Thu 6 Aug | Full grid on A100 | Complete grid, zero missing cells |
| 8 | Fri 7 Aug | Identical grid on L4 | Two grids over one config space |
| 9 | Sat 8 Aug | Plot both frontiers, mark membership, write the verdict paragraph | The crossing, or its documented absence |
| 10 | Sun 9 Aug | **Paper 1** | Finished draft |

### Headline figure
recall@5 versus p95 latency. One curve per card. Identical configuration points labelled on both. Frontier members marked. Looking for reranker configurations above the line on one card and below it on the other.

---

## 6. Measurement 02 — Ingestion

**Dates:** Tue 11 Aug → Mon 17 Aug (6 measure + 1 write)

### Question
At a fixed target ingestion throughput, is the budget better spent on systems-side optimisation or algorithm-side sampling?

### Hypothesis 2

**Amended 3 Aug 2026, before any data collection.** The original form predicted
that resolution reduction simply costs recall. Verifying the practitioner claim
it argues against (§4.2) showed the industry's own justification is a *threshold*
argument — models resize every frame to a fixed input size, so anything above
that size is free — and that the threshold varies from 224 to 1024 px across
models. The hypothesis now predicts the shape rather than the direction, which is
both more falsifiable and closer to what the mechanism implies.

> Systems-side ingestion optimisations are **recall-neutral**; sampling
> strategies are **recall-costly**. Resolution is filed as a systems knob but
> behaves as one **only above a threshold**: recall is flat from native
> resolution down to the embedder's native input size, then drops below it. The
> widely-repeated 360p advice is therefore safe for 224-input models and lossy
> for larger-input ones, and the recommendation never states which.

### Refuted if
Recall is flat across **every** resolution rung including those below the
embedder's input size, **or** it declines steadily from native with no threshold,
**or** any sampling strategy turns out recall-neutral.

### Knob table

| Family | Knob | Levels |
|---|---|---|
| Systems | `grab`/`retrieve` split | off / on |
| Systems | Decode path | CPU / NVDEC |
| Systems | Embedding batch size | 1 / 8 / 32 |
| Systems | Worker count | 1 / 2 / 4 |
| Systems | **Resolution** | native / 720p / 480p / 360p / 224p / **+ ≥2 rungs below the embedder's input size** |
| Algorithmic | Sampling | uniform / motion-gated / shot-aware |

Resolution is deliberately listed as a systems knob. Whether it behaves like one is the paper.

**The resolution ladder must straddle the embedder's input size.** The threshold
cannot be located from rungs that all sit on one side of it. Once the embedder is
chosen (`M02-INGESTION.md` §17), fix its native input size and add rungs below —
e.g. for a 224-input model, add 160p and 112p; for a 384-input model, 224p and
160p already straddle it. A ladder that stops at 224p on a 224-input model
measures nothing and produces a flat result for a mechanical reason.

### Corpus and ground truth
Lecture video corpus. Record hours, resolution, codec, GOP structure. **30 natural-language queries with hand-labelled ground-truth timestamps.** Recall measured against the deliberately-slow baseline, which is the ceiling.

### Metrics
recall@5 against timestamp ground truth · achieved ingestion FPS (wall clock, end to end) · per-stage profile (decode / sample / embed / index) · peak VRAM · GPU utilisation.

### Day plan

| Day | Date | Work | The number |
|---|---|---|---|
| 11 | Tue 11 Aug | `SCOPE.md`. Then the deliberately naive baseline: full decode, every frame, native resolution, serial embedding. Profile it | Baseline FPS + per-stage profile |
| 12 | Wed 12 Aug | The query set: 30 queries with ground-truth timestamps | recall@5 on the slow baseline |
| 13 | Thu 13 Aug | Systems knobs. Each measured for FPS **and recall** — measuring recall is the point, since the systems literature assumes it unaffected | FPS gain and recall delta per knob |
| 14 | Fri 14 Aug | The resolution ladder at fixed sampling | Recall and FPS at each rung |
| 15 | Sat 15 Aug | Sampling strategies at matched frame budgets | Recall cost per FPS gained |
| 16 | Sun 16 Aug | Joint frontier, coloured by family. Answer: does resolution cluster with systems or with sampling? | The frontier, and where motion gating falls |
| 17 | Mon 17 Aug | **Paper 2** | Finished draft |

### Headline figure
recall@5 versus achieved ingestion FPS. One point per configuration, coloured by knob family. The prediction is that one point breaks its own family.

### Coupling to OmniSight
Motion-gated sampling **is** OmniSight's Algorithm 1 (ROI-aware spatial-motion sampling). This measurement tells you empirically whether your own capstone's sampling design sits on the frontier or below it — a Phase B result obtained as a side effect. Cite the Phase A book for the algorithm specification.

---

## 7. Measurement 03 — Serving

**Dates:** Tue 18 Aug → Tue 25 Aug (**7 measure + 1 write — eight days, not seven**)

The extra day exists because TensorRT-LLM engines are architecture-specific and must be built separately per card, and because engine build failures are slow and uninformative.

### Question
Where does the vLLM → TensorRT-LLM crossover sit on the concurrency axis, and does the crossover point move with GPU tier?

### Hypothesis 3
vLLM wins at low concurrency; TensorRT-LLM wins above a crossover point; and **the crossover moves with GPU tier**, arriving at higher concurrency on the L4, because engine-level optimisation needs compute headroom the weaker card lacks.

### Refuted if
One framework leads at every concurrency level on both cards, **or** the crossover sits at the same concurrency on both.

### Sweep

| Axis | Levels |
|---|---|
| Framework | vLLM / TensorRT-LLM |
| Card | A100 / L4 |
| Concurrency | 1 / 2 / 4 / 8 / 16 / 32 / 64 / 128 |
| Precision | FP16 / FP8 / INT8 / INT4 (where the architecture permits) |

**Record every refusal.** Which precisions each card rejects is a result, not an obstacle — it is the practical half of the hardware-dependence thesis.

### Metrics
TTFT · TPOT · output tokens/sec · p50 and p95 · cost per million tokens (measured tokens/sec ÷ host hourly rate) · **engine build time** · quality check per precision so throughput gains are not silently bought with degradation.

### Day plan

| Day | Date | Work | The number |
|---|---|---|---|
| 18 | Tue 18 Aug | `SCOPE.md`. vLLM serving, OpenAI-compatible endpoint, health check, explicit warmup | Tokens/sec at concurrency 1, checked against a published figure |
| 19 | Wed 19 Aug | Load harness: concurrency sweep driver, TTFT/TPOT/tails, cost model | One clean curve, vLLM on A100 |
| 20 | Thu 20 Aug | TensorRT-LLM engine, A100. Budget the whole day | An engine that serves + build time |
| 21 | Fri 21 Aug | TensorRT-LLM engine, L4. Fresh build, not a copy | Second engine + refusal log |
| 22 | Sat 22 Aug | Precision ladder, both cards | Throughput and quality per precision, refusals included |
| 23 | Sun 23 Aug | Full concurrency sweep, four combinations | Four complete curves |
| 24 | Mon 24 Aug | Cost normalisation, locate crossovers, verdict | Crossover concurrency per card |
| 25 | Tue 25 Aug | **Paper 3** | Finished draft |

### Headline figure
Throughput per GPU-dollar-hour versus concurrency. Four curves (two frameworks × two cards). The prediction is two crossings at different concurrency levels, not one.

---

## 8. Apply gate — Monday 10 August

**Non-negotiable and independent of state.**

Applications go out on 10 August with whatever is committed. Measurement 01 need not have confirmed anything; it needs to exist, have a plot, and have a paper beside it.

**Why the date is fixed.** Graduation is February 2027. On 10 August the student runway is roughly six months; waiting for day 28 reduces it to five. Student positions are structured around retention, so runway is the binding eligibility constraint — and it shrinks faster than the portfolio grows. Hiring pipelines take three to six weeks regardless, so interviews land during Measurement 03 either way.

**Execution:**

- Three well-matched roles. Not a spray.
- Cover note order: the Paper 1 result → the RAG study ("David vs Goliath") → the MCP server design from OmniSight Phase A.
- State the February 2027 graduation date plainly rather than letting it be discovered.
- Twenty minutes, taken from the shallowest part of the day, never from build time.

---

## 9. Schedule

| Block | Dates | Days | Output |
|---|---|---|---|
| Block 0 — the instrument | Fri 31 Jul → Sun 2 Aug | 3 | Template, sweep runner, provenance, both cards verified |
| Measurement 01 — Retrieval | Mon 3 Aug → Sun 9 Aug | 7 | Two frontiers, verdict, **Paper 1** |
| **Apply gate** | **Mon 10 Aug** | 1 | Three applications sent |
| Measurement 02 — Ingestion | Tue 11 Aug → Mon 17 Aug | 7 | Joint frontier, verdict, **Paper 2** |
| Measurement 03 — Serving | Tue 18 Aug → Tue 25 Aug | 8 | Four curves, crossovers, **Paper 3** |
| Assembly | Wed 26 Aug → Fri 28 Aug | 3 | Three portfolio pages, **Paper 4**, one-command repo |

**Total: 28 working days + 1 apply day.**

### Block 0 detail

| Day | Date | Work | The number |
|---|---|---|---|
| 1 | Fri 31 Jul | Template (uv pyproject, Dockerfile, Makefile, pytest, ruff, `.env.example`) + `bench.py` emitting p50/p95/peak-VRAM as one JSON record | One metrics record from a dummy callable |
| 2 | Sat 1 Aug | Sweep runner: config grid in, one JSON line per cell out, resumable after crash. Plus `plot.py` | 12-cell dummy sweep completes and plots |
| 3 | Sun 2 Aug | Provenance stamping (GPU, driver, library versions, git SHA, seed). Rent both cards, run the dummy sweep on each | Two identical sweeps, distinguishable only by stamp |

**Block 0 is the one block worth writing by hand** — not for practice, but because every number claimed later inherits its trustworthiness, and you have to be able to defend it in an interview.

---

## 10. Paper structure — Papers 1 to 3

**No fixed page cap.** Amended 30 July 2026. The section budget below is a **starting point, not a ceiling** — roughly 4–6 pages and 2,500–3,500 words if nothing earns extra room, and longer where something does.

These are personal-site write-ups until Paper 4 is decided (Rule 8), so no venue is imposing a limit. The old cap was self-imposed discipline, and the discipline it protected still holds — what changes is that length is now decided section by section rather than in advance.

**The test a section must pass to grow.** Extra length is earned by *evidence*, never by explanation. A section may exceed its budget if the added words are one of:

- a result that does not fit the headline figure — a refusal log, a per-knob table, a failure mode worth recording
- a methodological detail a replicator would need and could not guess
- a threat to validity named honestly, with its consequence stated

It may **not** grow by restating the introduction in the conclusion, narrating what the reader is about to see, explaining background any reader of this venue already has, or hedging a finding across a paragraph that one sentence stated cleanly.

**The check.** On day 7, for each section that ran long, name the specific fact that justified the extra space. If you cannot name it in one sentence, cut back to the budget. Padding is still the first thing a reviewer notices in work from someone without a publication record — the risk did not go away, it just stopped being managed by a page count.

**Figures.** Still one headline figure per paper, and a second only if it says something the first cannot. Additional *tables* are cheap and count as evidence; additional figures are not, and usually mean the headline figure is doing too little.

### 10.1 What gets written when

**Amendment, 30 July 2026.** Day 7 is unchanged as a writing-only day. Days 1–6 now end with 15–30 minutes of writing, taken from the end of the day, never from the front.

The split is by **dependency on the verdict**. Sections that do not depend on results are written the day the thing they describe is built — Method written on the day you built the harness is accurate; Method reconstructed on day 7 is a guess. Sections that interpret results are written only on day 7, after the sweep is complete.

| Section | Written | Why |
|---|---|---|
| Related work + gap sentence | Day 1, from §4 of this file | Independent of data. Already drafted here |
| Method — system under test, config grid | Day 1, alongside `SCOPE.md` | The grid is frozen on day 1 anyway |
| Method — corpus and gold set construction | The day the gold set is built | Labelling procedure must be described while you remember it |
| Method — harness, metrics, provenance fields | The day the instrumentation is written | Same reason |
| `LOG.md` entry | End of every day, 10 minutes | Raw, not prose. See below |
| Abstract, Results, Threats, Conclusion | **Day 7 only** | These require the verdict |

**The daily `LOG.md` entry.** One block per day, committed with that day's work: what ran, the number it produced, anything surprising, anything broken and how it was worked around. Raw notes, not paragraphs. This is a lab notebook, not a draft — its job is to feed Threats to Validity, which is the section a reviewer uses to decide whether to trust you and the one that is impossible to reconstruct a week later.

**The rule that protects this.** No sentence interpreting a result may be written before the sweep that produces it is complete. Logging a number is fine; explaining what it means is not. Writing a verdict on day 4 commits you to a story before the data has finished arriving, which is Rule 1 failing through the back door.

**What day 7 becomes.** Assembly plus the four verdict-dependent sections — roughly half the previous load, on better source material. If day 7 still feels full, the front-matter sections were skipped during the week.

### Section budget

| Section | Words | Contents |
|---|---|---|
| Abstract | 150 | Lead with the number, not the setup |
| 1. Introduction | 400 | Practitioner problem → question → contributions as three bullets |
| 2. Related work | 350 | One or two paragraphs, ending in the gap sentence |
| 3. Method | 700 | System under test, config space table, harness, hardware, metrics |
| 4. Results | 800 | One headline figure, one or two tables, the verdict |
| 5. Threats to validity | 250 | Written honestly — where a reviewer decides whether to trust you |
| 6. Conclusion | 150 | What a practitioner should do differently |
| References | — | 10–15 entries |

Word counts are the starting allocation. Results and Threats are the two sections most likely to legitimately outgrow theirs, because both hold evidence; Introduction and Conclusion almost never do, because both hold framing.

### Paper 1 — Retrieval

**Title:** *Does the Pareto-optimal RAG configuration transfer across GPU tiers?*

- **Introduction** — Configuration-search papers publish a recommended pipeline benchmarked on hardware most readers do not have. Whether the recommendation survives the move is untested.
- **Related work** — syftr, RAG-Stack, RAGSmith, METIS, cost–latency–quality benchmarks. One sentence each, then the gap sentence from §4.1.
- **Method** — Corpus and chunk count. The grid as a table with cell count. Gold set construction: 30 queries, hand-labelled chunk IDs, one annotator, labelling procedure stated explicitly. Harness: per-stage timing, p50/p95, peak VRAM, provenance fields. Hardware with driver and library versions.
- **Results** — Figure 1: recall@5 vs p95 latency, both cards, frontier membership marked. Table: frontier membership per card. Then knob by knob — which optimal values changed and which did not. Verdict paragraph.
- **Threats** — One corpus. One embedding family. Two cards is not a continuum. Single seed. Single annotator on the gold set.

### Paper 2 — Ingestion

**Title:** *Which video ingestion optimisations are free? The recall cost of throughput gains*

- **Introduction** — The 360p/30fps advice circulates widely with no recall number attached.
- **Related work** — **Two paragraphs, and the two-paragraph structure is itself the argument.** First: systems work (CoVA's compressed-domain filtering, the grab/retrieve case study). Second: retrieval work (frame-sampling comparison for video RAG, shot-aware sampling). Close with the gap sentence from §4.2.
- **Method** — Lecture corpus: hours, resolution, codec, GOP structure. Query set: 30 queries with timestamp ground truth. Knob table with each row tagged systems or algorithmic. Harness and hardware.
- **Results** — Figure 1: recall@5 vs achieved ingestion FPS, coloured by family. Table: FPS gain and recall delta per knob. Answer the resolution question directly. Report where motion gating lands.
- **Threats** — **Lecture video is low-motion, so motion gating will likely underperform relative to surveillance footage. State this explicitly.** It is the obvious objection, and naming it first is worth more than the result it qualifies. Also: one corpus, one embedding model, single annotator.

### Paper 3 — Serving

**Title:** *Where does the vLLM / TensorRT-LLM crossover sit, and does it move with the GPU?*

- **Introduction** — Framework choice is currently made from vendor benchmarks reporting single operating points.
- **Related work** — APEX4 and AnyBCQ, positioned precisely as answering the *kernel-level* version of this question, with this work at the *framework* level. QServe for the co-design angle. Naming them precisely is what prevents a reviewer assuming you missed them.
- **Method** — Model, prompt-length distribution, concurrency levels, precision ladder with the support-matrix constraint stated up front. Cost model: measured tokens/sec + host hourly rate → dollars per million tokens. Engine build time as a recorded quantity.
- **Results** — Figure 1: four curves, throughput per GPU-dollar-hour vs concurrency. Table: observed precision support including refusals. Table: engine build times. Crossover points per card.
- **Threats** — **Rented hosts have noisy neighbours; report variance across repeated runs.** One model. Spot prices move over time. Single seed. Concurrency sweep is synthetic load, not production traffic.

---

## 11. Paper structure — Paper 4

Different shape. Paper 4 is **not** a summary of the first three — if it were, it would have no reason to exist. It is the only paper with three settings to generalise from, which is what makes it the one worth submitting.

Target 10–14 pages.

| § | Section | Notes |
|---|---|---|
| — | Abstract | The general claim, then the three settings as evidence |
| 1 | Introduction | The general problem: published optimal configurations carry unstated hardware preconditions |
| 2 | Related work | ~1 page, three subsections mirroring §4.1–4.3 |
| 3 | **A taxonomy of configuration knobs** | Hardware-invariant vs hardware-dependent, with a stated criterion for the split. This is a contribution in its own right |
| 4 | Method | Three studies, **one shared harness**. The shared instrument is what makes them comparable and is a methodological contribution |
| 5 | Results A — retrieval | One compressed page, findings only |
| 6 | Results B — ingestion | One compressed page |
| 7 | Results C — serving | One compressed page |
| 8 | **Cross-cutting analysis** | **The paper.** See below |
| 9 | Practitioner guidance | What to re-measure on a hardware change; what can safely be copied |
| 10 | Threats to validity | Full section, not a paragraph |
| 11 | Conclusion | |

### Section 8 is the contribution

Everything else is scaffolding around it. It must do three things:

1. **Report which knob families moved and which did not**, across all three settings, in one table.
2. **Attempt a predictor.** Is there a hardware property that forecasts which knobs will shift? APEX4 established that the tensor-core-to-CUDA-core ratio does this at kernel level. The configuration-level analogue may be the memory-bandwidth ratio, or the compute-to-bandwidth ratio. **Even a negative answer is publishable** — "no simple hardware statistic predicts configuration sensitivity" is a useful finding.
3. **State the general claim and its limits.** Two cards and three settings supports a claim about existence, not about magnitude. Say so.

If section 8 is skipped or thin, Paper 4 is three papers stapled together and should not be submitted.

### Venue shape
A systems-for-ML workshop or an empirical/reproducibility track is the right target. Not a top-tier main conference — the evidence base is two cards. Aim where careful measurement of a practical question is the accepted contribution.

---

## 12. Portfolio pages

One page per paper, written on 26 August.

- **Titled with the finding, not the project name.** "Reranking flips from optimal to dominated when the GPU changes" — not "RAG Benchmark Project".
- Headline plot first, above any prose.
- Two paragraphs: what was measured, what came out.
- Links: PDF, repo, reproduction command.
- One shared closing line linking the three, so the common thread reads as consistency rather than repetition.

Three pages led by three different findings, plus one paper making the general claim, is a coherent portfolio around a single specialisation. Three unrelated projects would not be.

---

## 13. Study debt ledger

**The deliberate trade:** build fast now, understand deeply afterwards at a sustainable pace.

**The guard against that becoming permanent:** each paper's Method section forces an explanation of every parameter chosen. Writing Method *is* the minimum viable understanding — a configuration you do not follow cannot be described. This is why the writing days are non-optional even though the deep pass is deferred.

The deep pass runs against this list after 28 August, at whatever pace works.

### From Measurement 01
- Embedding model internals — what the pooling actually does
- HNSW parameters: `M`, `ef_construction`, `ef_search`, and how they trade recall against latency
- Cross-encoder architecture, and why it costs what it costs
- Why recall@k and MRR disagree, and which one matters for which use case

### From Measurement 02
- H.264 GOP structure and keyframe placement
- The NVDEC pipeline, and where host↔device transfers dominate
- Motion vectors in the compressed domain
- Why lower resolution hurts some queries and not others

### From Measurement 03
- PagedAttention and KV-cache paging
- Continuous batching versus static batching
- What engine compilation actually specialises, and why it is architecture-specific
- Quantization arithmetic, and where accuracy leaks

---

## 14. Failure modes and responses

| If | Then |
|---|---|
| A grid will not finish in the day | Cut levels, not cards. Report the reduced grid honestly |
| The hypothesis is refuted early | Good. Finish the sweep anyway — a clean refutation needs the full grid |
| An engine build fails all day | Record the failure modes; they belong in the paper. Fall back to fewer precisions |
| A card is unavailable on the rental market | Wait for it rather than substituting silently. Card identity is load-bearing |
| Slipping more than two days | Cut Measurement 03 to vLLM-only across both cards and reframe Paper 3 around precision support and cost, not framework crossover |
| The 10 Aug gate arrives with Paper 1 unfinished | Apply anyway. That is the whole point of the gate |

---

## 15. Citations to verify before writing

Confirm exact identifiers and BibTeX from the primary sources. **Do not cite from this list without checking** — names and findings here are accurate, identifiers need verification.

**Confirmed (from the OmniSight Phase A book):**

- Lewis et al. (2020), *Retrieval-augmented generation for knowledge-intensive NLP tasks*, NeurIPS — arXiv 2005.11401
- Malkov & Yashunin (2020), *HNSW*, IEEE TPAMI 42(4) — arXiv 1603.09320
- Radford et al. (2021), *Learning transferable visual models from natural language supervision*, ICML — arXiv 2103.00020
- Tschannen et al. (2025), *SigLIP 2* — arXiv 2502.14786
- Wang et al. (2024), *InternVideo2*, ECCV — arXiv 2403.15377
- Bai et al. (2025), *Qwen2.5-VL technical report* — arXiv 2502.13923
- Vaswani et al. (2017), *Attention is all you need*, NeurIPS — arXiv 1706.03762

**To locate (names and findings verified, identifiers not):**

- syftr — multi-objective RAG configuration search
- RAG-Stack — joint algorithm–system Pareto frontier
- RAGSmith — evolutionary pipeline search
- METIS — retrieval-augmented pipeline exploration
- CoVA — compressed-domain video analytics, ~4.8× throughput
- The 2025 grab/retrieve industrial ingestion case study
- Frame-sampling comparison for video RAG
- Shot-aware sampling; ProCLIP

**Confirmed 3 Aug 2026 (primary source fetched directly):**

- Coactive, *Unlocking Video Insights: How an Optimized Video Ingestion Pipeline Transforms Content Discovery*. https://www.coactive.ai/blog/optimized-video-ingestion-pipeline — source of the 360p/30fps recommendation, the fixed-input-resolution justification, and the 224–1024 px range. This is Paper 2's practitioner-problem citation (§4.2); cite it as industry documentation, not as peer-reviewed work.
- APEX4 (Jun 2026) — tensor-core-to-CUDA-core ratio, platform-dependent W4A4
- ~~AnyBCQ — speedup preservation across A100/H100~~ **VERIFIED 5 Aug 2026: claim was wrong, see §4.3.** Correct summary: multi-precision binary-coded quantization, arXiv:2510.10467, ICLR 2026
- QServe — quantized serving co-design
- TensorRT-LLM support matrix (primary documentation, for the SM75 constraint)
- vLLM PagedAttention paper

---

## 16. Prior work of your own to cite or reference

- **"David vs Goliath"** — agentic versus naive RAG across model scales on HotpotQA. Interaction effect: agentic scaffolding helped the larger model and hurt Llama-3.1-8B. Relevant to Paper 1 and to Paper 4's framing, since it is the same *shape* of result: the optimal configuration depends on what you are running it on. That paper found model-dependence; this programme finds hardware-dependence.
- **OmniSight Phase A** (Capstone 26-2-D-17, Braude, with Cyrine Fahoum, advised by Dr. Reuven Cohen) — Algorithm 1, ROI-aware spatial-motion sampling, is the motion-gating strategy evaluated in Measurement 02. Cite the Phase A book for the specification.

---

*Last updated 30 July 2026. Amend this file rather than keeping decisions in your head.*
