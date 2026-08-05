# LOG — Measurement 03: Serving

Raw notes, not prose. One block per day, committed with that day's work
(`REFERENCE.md` §10.1). Day numbers are block-relative, matching
`Retrieval`/`Ingestion`'s convention — `M03-SERVING.md`'s own section
numbering calls this block's days "Day 18"–"Day 25" (the programme's
original 28-day schedule); this file uses "Day 1"–"Day 8" instead and notes
the cross-reference where it matters.

---

## Day 1 — Wed 5 Aug 2026

Started immediately after Measurement 02's halt, per `Ingestion/LOG.md` →
*Day 3 — Measurement 02 halted*, which names Serving as next: its ground
truth is machine-generated (TTFT/TPOT/throughput read off a clock), so the
annotation-labour failure that stopped M02 cannot recur here. Not waiting for
the block's originally-planned calendar date (18 Aug) — `REFERENCE.md`'s
schedule is being compressed in response to M02's early stop, and there is
no reason to sit idle between blocks.

**Environment split decided first, before any code.** Dev machine (this one)
has no CUDA GPU — `torch.cuda.is_available() == False`, same as `Retrieval`
Day 1 and `Ingestion` throughout. Real runs happen on Colab Pro, against the
rented A100/L4. New wrinkle specific to this block: **vLLM has no Windows
wheel at all**, and confirmed today that **neither does TensorRT-LLM at any
version** — `uv pip install --dry-run tensorrt-llm` (plain PyPI) fails
because the PyPI entry is a stub that redirects to
`https://pypi.nvidia.com/`; retrying with
`--extra-index-url https://pypi.nvidia.com/` resolves real versions but
every one of them publishes wheels only for `linux_x86_64` /
`linux_aarch64` — none for any Windows target, at any Python ABI. So unlike
`Retrieval`/`Ingestion` (where `torch`/`transformers` at least *install* on
Windows via a CPU-only wheel redirect and produce a real, if slow, CPU smoke
test), the serving frameworks themselves cannot be imported here even for a
structural test. Consequence: `pyproject.toml` does not list vLLM or
TensorRT-LLM as dependencies at all (they also pin `torch`/`transformers`
versions that conflict with what this project's own tooling needs — vLLM
0.26.0 wants `torch==2.11.0`, this project wants `torch>=2.13.0` and
`transformers>=5`); both are installed in their own environment on the GPU
host per `configs/serve-env.md`. Everything in `serving/` (`pins.py`,
`provenance.py`, `prompts.py`, `quality.py`, `config.py`) is plain
Python/torch/transformers/huggingface_hub and is fully unit-tested here on
CPU (27 tests, `uv run pytest`, all passing).

**Open decision §18 — hypothesis direction — settled with Ahmad before
`SCOPE.md` was written.** Chose **vLLM-low / TensorRT-LLM-high**, the option
`REFERENCE.md` §7 already commits to and `M03-SERVING.md` §1.1 lists first.
Reasoning: at concurrency 1 there is nothing to batch and decode is
bandwidth-bound, so neither framework can compile its way past the memory
wall — they should be close. TensorRT-LLM's fused kernels and in-flight
batching are load-time optimisations and should show up under load, not at
concurrency 1. The reverse direction (TRT leads at concurrency 1 because
compiled kernels shorten every request regardless of load; vLLM's
PagedAttention is a memory-packing advantage worth nothing until concurrency
is high) was equally arguable a priori and is explicitly **not**
pre-registered — per §1.1 only one direction can be committed, and it will
not be revised after seeing data. Written into `SCOPE.md` before any other
Day 1 decision.

**Model chosen and pinned.** `Qwen/Qwen2.5-7B-Instruct`, revision
`a09a35458c702b33eeacc393d103063234e8bc28` (resolved via
`serving/pins.py`, `HfApi().model_info().sha`, stored in
`configs/model_pins.yaml`). Considered `meta-llama/Llama-3.1-8B-Instruct`
first (§18's "7–8B instruct model" working assumption) but rejected it after
checking two things directly rather than assuming:

1. **It is gated.** `AutoConfig.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")`
   returned a live 401 ("You are trying to access a gated repo") with no
   `HF_TOKEN` set. A gated model ties reproduction to one account's license
   acceptance — an avoidable dependency for a study that isn't about the
   model at all.
2. **The L4 24 GB arithmetic is worse than Qwen2.5-7B's.** Fetched both
   configs from the Hub (`AutoConfig`, no weights downloaded, works fine on
   CPU): Llama-3.1-8B has 8 KV heads (`hidden_size=4096, layers=32`), Qwen2.5
   -7B has only 4 (`hidden_size=3584, layers=28`) via more aggressive GQA.
   KV cache per token: Llama-3.1-8B ≈ 128 KiB, Qwen2.5-7B ≈ 56 KiB — roughly
   half. At concurrency 128 and 640 tokens/request (512 in + 128 out):
   - Llama-3.1-8B: weights ~16 GiB (bf16) + KV cache ~10.5 GiB ≈ **26.5 GiB
     — already over the L4's 24 GiB before activations or framework
     overhead**, i.e. concurrency 128 would not fit at all.
   - Qwen2.5-7B: weights ~14.2 GiB + KV cache ~4.4 GiB ≈ **18.6 GiB**,
     leaving ~5.4 GiB of headroom (~2–3 GiB against vLLM's typical
     `gpu_memory_utilization≈0.85–0.90` effective budget). Tight, not
     comfortable, but concurrency 128 is at least plausible on the L4.

   Full arithmetic, with the exact config numbers, lives in
   `serving/config.py` as a comment next to the constants it justifies
   rather than only here — so anyone reading the sweep code sees why 640
   tokens and this model were chosen, not just that they were.

   On the A100 (40 GiB) neither model is the constraint — ~21–24 GiB of
   headroom either way. The L4 is where this decision actually bites, which
   is exactly the asymmetry §2.3 predicts (the cheaper card is the tighter
   one) — worth restating in §3.1 of `PAPER.md` as it's written today.

**`prompts.py` frozen.** 8 topically-varied prompts (systems explanation,
code, arithmetic reasoning, incident writing, product copy — varied so the
Day 22 quality gate has real content, not repeated noise), each padded to
exactly 512 input tokens under the pinned model's own tokenizer. Frozen to
`data/prompts.json`, `prompts_sha = f4e86ac282ad6b4fbc03e97a41dd4255eea489a9`.

**One real bug caught while building this, worth recording (M01/M02-style
"two real bugs" note):** the first padding implementation concatenated
*token-id lists* (seed ids + tiled filler ids), sliced to exactly 512, then
called `tokenizer.decode()`. Re-tokenizing the decoded text came back at
**504 tokens, not 512, for all 8 prompts** — not one boundary artifact but a
systematic one. Root cause: BPE merges across the id-concatenation
boundary — e.g. a sentence-ending `"."` token immediately followed by a
`"For"` token (no leading space, since `"For"` was tokenized as the *start*
of the filler string) decodes to `"understand.For"` with no space, which
re-tokenizes as a single merged token instead of two. Every filler-tile
boundary in a padded prompt caused one such merge, and with ~8 tiles needed
to pad a short seed out to 512 tokens, that is exactly the observed -8
shortfall — confirmed by walking the id/re-encode diff directly
(`ids[26]` was the first divergence: `[..., 13, 2461, ...]` i.e. `"."`,
`"For"` had become a single `26676` token after the round-trip). Fixed by
tiling the *filler string* (with an explicit space) rather than concatenating
id lists, tokenizing the whole thing once, and then closing an explicit
convergence loop — `decode(ids[:target])`, re-tokenize, adjust `target` by
the observed delta, repeat — since even that fix doesn't *guarantee*
`decode(ids[:n])` re-encodes to exactly `n` tokens at the single remaining
cut point. Converged in 0 retries for all 8 real prompts, but the retry path
itself is unit-tested against an adversarial fake tokenizer that reproduces
the merge deterministically (`tests/test_prompts.py`, `_FlakyTokenizer`) —
both the "converges on retry" and "raises after exhausting the retry budget
rather than looping forever" cases are covered, not just the case that
happened to work on Qwen's real tokenizer.

**TensorRT-LLM Day 18 spike (§6) — not run today, cannot be, and that
non-run is itself the useful Day 1 fact.** §6 asks whether the toolchain
installs and builds *anything* on this runtime within 30 minutes — the
runtime for that question is the GPU host, not the dev box, and the dev-box
finding above (`tensorrt-llm`: zero Windows wheels, any version) confirms
there is no partial answer obtainable here either — it fails at `pip
install`, before any build step. `scripts/trtllm_spike.sh` is written
(timeboxed with a wall-clock budget check between each step, smallest
available model = `gpt2`, not the study model) and ready to run in a Colab
cell against the A100 runtime. **Genuinely timeboxed and run next session on
Colab; result goes here as its own dated entry, not backfilled into today's.**

**vLLM serving + concurrency-1 sanity number — same situation.**
`scripts/serve_vllm.py` is written (health check, fixed warmup, the
`stream_completion`/TTFT-TPOT formula from §4.1) and refuses to run outside
a Linux+CUDA host with a clear message (`_require_gpu_host()`) rather than
failing on a missing `vllm` import. **The actual concurrency-1 number is
pending the first Colab session** — recorded here once it exists, not
estimated in its place.

**Build:** `SCOPE.md` · `serving/config.py`, `serving/pins.py`,
`serving/provenance.py`, `serving/prompts.py`, `serving/quality.py` ·
`scripts/serve_vllm.py` · `scripts/client.py` (skeleton, per the Day 18 build
list — full closed-loop N-worker driver is Day 2/"Day 19") ·
`scripts/trtllm_spike.sh` · `configs/serve-env.md` · `configs/model_pins.yaml`
· `data/prompts.json` · 27 passing unit tests (`tests/test_pins.py`,
`tests/test_provenance.py`, `tests/test_prompts.py`, `tests/test_quality.py`)
· `pyproject.toml`/`uv.lock` (vLLM/TensorRT-LLM deliberately excluded, see
above) · git repo initialised.

**Write:** `PAPER.md` — §1 hypothesis paragraph, §2 Related work in full
(from `REFERENCE.md` §4.3), §3.1 System under test (model choice with the
arithmetic, cards, frameworks, prompt set), §3.2 sweep table and held-fixed
controls.

**Next session (Colab, GPU host):** run `scripts/trtllm_spike.sh` on the
A100 runtime, timeboxed 30 minutes, record wall time and outcome verbatim —
success or the exact failure string. Launch `scripts/serve_vllm.py --card
A100-SXM4-40GB --precision fp16`, record the health-check time, warmup
behaviour, and the concurrency-1 sanity number (tokens/sec, TTFT, TPOT),
checked by eye against a published figure for a 7B-class model on an A100.

`[outcome: Day 1 build complete on the CPU dev box; the two GPU-dependent
Day-18 deliverables (TensorRT-LLM spike outcome, concurrency-1 sanity
number) are queued for the next Colab session, not faked or estimated here]`

---

## Day 2 — Wed 5 Aug 2026

M03-SERVING.md calls this "Day 19": the load harness, and proving it is not
the bottleneck. Unlike Day 1's GPU-dependent deliverables, **the entire
§4.2 validity gate turned out to be runnable on the CPU dev box** —
`mock_server.py` has no real compute behind it, so the harness could
actually be proven today rather than only written and queued for Colab.

**A real bug, found and fixed, worth recording in the M01/M02 style.**
`client.py` and `mock_server.py` worked individually but a full-stream test
through `mock_server.py` showed `n_tokens` correct (128, matching the
forced `min_tokens`) but wall time far too short — a single request with a
configured ~1.32s of total sleep completed in ~0.17s. Isolated with a
minimal repro (`asyncio.sleep(0.01)` in a loop, interleaved with
`aiohttp` `StreamResponse.write()` calls) down to: **Windows' default
`ProactorEventLoop` silently collapses short (`0.01s`) `asyncio.sleep()`
calls to near-zero when they're interleaved with stream writes.** A plain
`for i in range(10): await asyncio.sleep(0.01)` loop with no aiohttp
involved timed correctly (~0.157s for 10×10ms); the same 10 sleeps
interleaved with `resp.write()` calls in an aiohttp handler completed in
~0.004s. Confirmed the fix by switching to
`asyncio.WindowsSelectorEventLoopPolicy()` on `sys.platform == "win32"`
before `web.run_app` — same handler, same sleep durations, correct timing
(~0.156s for the same 10×10ms loop). Applied only under the `win32` guard in
`mock_server.py`; Colab (Linux) never hits this and keeps the default loop.
This is exactly the kind of thing that would have silently invalidated
today's validity-gate numbers if it had gone unnoticed — a "ceiling" that
completes 8x too fast is not a client that's keeping up, it's a server that
isn't actually pacing itself.

**The validity gate, run for real:** `scripts/mock_server.py` + all 8
concurrency levels (`scripts/sweep.py --framework mock --card mock`, 8s
window, 1 repeat, 8 total warmup requests distributed per §4.3) +
`scripts/validate_client.py`. Raw sweep in `data/mock_validity_gate.jsonl`,
full write-up in `results/harness_validation.md`.

- Achieved throughput scales **linearly with concurrency across the entire
  1→128 range** — ratio to `concurrency × (throughput at concurrency 1)`
  stays within **0.93–1.06** at every level, with no downward trend
  approaching 128. That flatness, not a match to any absolute number, is
  the actual validity signature: a client acting as the bottleneck would
  show the ratio *falling* as concurrency rises, not sitting at a constant
  level all the way to the top of the axis.
- Ratio to the mock server's own *theoretical* ceiling (computed from its
  configured 50ms TTFT + 10ms/token constants) sits around **0.57–0.64 at
  every level** — flat, not degrading, and explained by a second, smaller
  finding: even after the ProactorEventLoop fix, Windows' asyncio timer
  granularity still inflates the mock server's configured 10ms inter-token
  sleep to an *actual* ~16–17ms (measured directly: a 10-sleep loop
  requesting 100ms total took ~157ms). That's the server's own timing
  fidelity on this OS, not a client artifact — it moves the theoretical
  ceiling, not the linear-scaling check, which is why the gate is reported
  against both numbers rather than one.
- **Client CPU utilisation never exceeded 51% (mean) even at concurrency
  128** (`psutil.Process().cpu_percent()`, sampled every 0.5s throughout
  each window) — nowhere near pegged, consistent with the linear-scaling
  result.

**Result: PASS.** `scripts/client.py`'s closed-loop design (`aiohttp`,
`TCPConnector(limit=0)` — aiohttp's default 100-connection cap would have
silently truncated the top of the concurrency axis, worth noting since it
is exactly the kind of thing that fails quietly) holds 128 concurrent
in-flight streams without becoming the bottleneck itself. This is the §4.2
gate that has to pass before any real server number gets trusted, and it
now has.

**Warmup-count interpretation, made explicit rather than left implicit.**
§18's "32 requests" working assumption doesn't state whether that's 32 per
worker or 32 total. Read it as **32 total**, distributed round-robin across
the level's workers (`sweep._warmup_count_per_worker`, floored to a minimum
of 1 per worker) — the alternative (32 *per worker*) would mean 4,096
discarded warmup requests at concurrency 128, which at a real ~1.3s/request
would be over an hour of warmup before a single measured request. Flagging
this interpretation explicitly rather than silently picking one, since §18
itself says this number needs confirming against a real curve — that
confirmation is queued for the Colab session along with the concurrency-1
sanity number.

**Build:** `serving/record.py` (schema assembly, percentiles, goodput,
JSONL read/write) · `scripts/mock_server.py` (with the event-loop-policy
fix) · `scripts/client.py` completed (full closed-loop N-worker driver,
`run_warmup`, `run_measurement_window`, client CPU sampling) ·
`scripts/sweep.py` (concurrency-ladder driver, framework-agnostic — same
code will drive vLLM/TensorRT-LLM later) · `scripts/validate_client.py` ·
`results/harness_validation.md` · `data/mock_validity_gate.jsonl` · 23 new
tests (`tests/test_mock_server.py`, `tests/test_record.py`,
`tests/test_client.py`, `tests/test_sweep.py`) — 50 total, all passing on
CPU.

**Write:** `PAPER.md` §3.3 Measurement protocol (closed-loop rationale,
TTFT/TPOT formulas with the `-1` note, warmup policy, window/repeat
rationale, p50/p95 rationale), §3.4 Harness validation (the mock-server
result, stated as numbers).

**Still queued for Colab (unchanged from Day 1):** the TensorRT-LLM 30
-minute spike (`scripts/trtllm_spike.sh`), the vLLM concurrency-1 sanity
number (`scripts/serve_vllm.py`), and now also **the first real curve**
(vLLM on A100, FP16, all 8 concurrency levels via `scripts/sweep.py
--framework vllm --card A100-SXM4-40GB`) — the harness that will produce it
is now built, tested, and proven not to be the bottleneck, but running it
against a real model still needs the rented GPU.

`[outcome: §4.2 validity gate PASSED on the CPU dev box, with two real bugs
found and fixed along the way (ProactorEventLoop sleep collapse; aiohttp's
default 100-connection cap). The harness is ready; the first real curve is
still queued for Colab, not faked or estimated here]`
