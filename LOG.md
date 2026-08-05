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

---

## Day 3 — Wed 5 Aug 2026

M03-SERVING.md calls this "Day 20": TensorRT-LLM engine, A100, budget the
whole day. **Nothing in that day's Build-and-run list can execute on the
CPU dev box at all** — engine building needs TensorRT-LLM, which (LOG.md
Day 1) doesn't even install on Windows, let alone build without a CUDA
device. Everything today is code written and structurally tested, queued
for the Colab session, same honesty as Day 1's vLLM sanity number.

**Refactor before writing the new framework.** `serve_vllm.py`'s
health-check/warmup/sanity-check code and its `stream_completion` would
otherwise have been copy-pasted into a new `serve_trtllm.py` — a third
implementation of the §4.1 TTFT/TPOT formula (after `client.py`'s async
version and `serve_vllm.py`'s sync one), which is exactly the kind of drift
this project keeps flagging as a risk. Pulled the shared pieces into
`serving/serve_common.py` first; `serve_vllm.py` now only owns its
framework-specific `launch()` and `_DTYPE_FLAGS`.

**A second real bug, this one in testing the refactor.** The first pass at
`tests/test_serve_common.py` ran `mock_server.py` in-process
(`aiohttp.test_utils.TestServer`) and called `serve_common.stream_completion`
(a synchronous `requests` call) via `asyncio.to_thread` from inside the same
event loop. Timing came back wrong in a new way: TTFT measured at **2.14s**
against a server configured for ~0.05s. Isolated by timing a plain
`requests` call with *no* event loop involved at all — 0.078s, correct —
which pinned the cause on the interaction itself: `requests`, called via
`to_thread`, contending with an active `WindowsSelectorEventLoopPolicy` loop
in the same process. (This is a different failure mode from Day 2's
Proactor-sleep-collapse bug, and a good reminder that "the fix from
yesterday" doesn't mean "no more event-loop surprises today.") Fixed by
switching the affected tests to a **real subprocess** `mock_server.py`
instead of an in-process test server — simpler, and closer to how
`serve_vllm.py`/`serve_trtllm.py` actually talk to a server in practice (a
separate process, always). Re-ran: 7/7 passing, no stall, ~7s total instead
of 17s.

**`serving/build_log.py`** — engine-build timing/RAM/VRAM logging, generic
over the command run. Peak host RAM (`psutil`, whole-system, since a build
spawns its own subprocesses and per-process RSS would undercount) and peak
VRAM (`nvidia-smi`, polled every second in a background thread; returns
`None` cleanly on this GPU-less dev box, same spirit as
`provenance.stamp()`'s "none (CPU)"). Records both successful and failed
build steps, per §6's "record every failed attempt, not just the
successful one." 7 tests, including one that confirms `_nvidia_smi_used_mb()`
genuinely hits its except-and-return-`None` path on this machine rather
than a mocked one.

**`scripts/build_trtllm_engine.py`** — the Day 20 build script itself.
Attempts a direct HF-checkpoint build first, falls back to a per-model
convert-then-build flow on failure, and records *which* path worked (or
both failures) via `build_log`. Persists to Drive and verifies the copy by
file-count/byte-size comparison, not a full content hash — engine files run
into the gigabytes, and size/count already catches a partial Drive write at
a fraction of the I/O cost. The two conversion paths and their exact CLI
flags are flagged provisional in the file's own docstring, same as
`serve_vllm.py`'s `_DTYPE_FLAGS` — confirmed against the real installed
version on Colab, not assumed here. `_engine_id` and `persist_to_drive` are
unit-tested against `tmp_path` directories (5 tests); the build steps
themselves need `trtllm-build` and are not — there is nothing to fake there
that wouldn't just be lying about what was tested.

**`scripts/serve_trtllm.py`** — the TensorRT-LLM twin of `serve_vllm.py`,
now genuinely thin thanks to the refactor: only `launch()` (the
`trtllm-serve` command) and the GPU-host guard are framework-specific.
Defaults to port 8001 (vLLM stays on 8000) so both can run side by side —
required for the parity check below.

**`scripts/check_parity.py`** — same prompt set, greedy, exact match
required (no tolerance band, unlike the Day 22 quality gate — these are two
servers both claiming to run the identical FP16 model, so anything short of
exact means one of them isn't). Reuses `serving/quality.exact_match_rate`
rather than a second comparison implementation. Tested for real, not just
structurally: `mock_server.py` got a `--token-text` flag so two instances
can be started that deliberately disagree (`"x"` vs `"y"`), and
`test_check_parity.py` confirms both the agree→PASS and diverge→FAIL paths
against genuinely different server output, plus `main()`'s exit codes. One
self-inflicted bug caught while writing these tests: an early version tried
to monkeypatch `shutil.copytree` with a replacement that called
`import shutil; shutil.copytree(...)` internally — since `shutil` is one
shared module object, that replacement was calling *itself*, producing a
`RecursionError`. Fixed by stubbing the copy out entirely and pre-seeding
the destination directory instead of trying to wrap the real function.

**Cleanup:** `mock_server.py`'s app-storage key switched from a plain string
(`app["token_text"]`) to `web.AppKey`, clearing an aiohttp deprecation
warning that showed up once `--token-text` was added and multiple app
instances existed side by side in the same test run.

**Build:** `serving/serve_common.py` · `serving/build_log.py` ·
`scripts/serve_trtllm.py` · `scripts/build_trtllm_engine.py` ·
`scripts/check_parity.py` · `scripts/serve_vllm.py` (refactored onto
`serve_common`) · `scripts/mock_server.py` (`--token-text`, `AppKey` fix) ·
17 new tests (`test_serve_common.py`, `test_build_log.py`,
`test_build_trtllm_engine.py`, `test_check_parity.py`) — 74 total, all
passing on CPU, no warnings.

**Write:** `PAPER.md` §3.5 Engine build procedure (both conversion paths,
the persistence/verification design, the parity-check rule), §4.1 the
build-time table skeleton (columns fixed, rows pending Colab).

**Queued for Colab (carried forward, plus today's additions):** the
TensorRT-LLM 30-minute spike, the vLLM concurrency-1 sanity number, the
first real vLLM/A100 curve (all since Day 1/2) — and now also the actual
A100 TensorRT-LLM engine build, its Drive persistence and verification, and
the vLLM/TensorRT-LLM parity check for real. All of today's scripts are
written, reviewed, and tested against everything that doesn't require a
GPU; none of them have touched a real engine build yet.

`[outcome: serve_common refactor done (one TTFT/TPOT implementation, not
three); build_log, build_trtllm_engine, serve_trtllm, and check_parity all
written and tested against everything CPU-testable; one new event-loop bug
found and fixed (requests+to_thread+Selector-loop contention) plus one
self-inflicted test bug (shutil monkeypatch recursion); the actual A100
engine build is still queued for Colab, not faked or estimated here]`

---

## Day 4 — Wed 5 Aug 2026

M03-SERVING.md calls this "Day 21": TensorRT-LLM engine, L4, fresh build,
and the §6 hard stop. Everything this day's Build-and-run list asks for —
an L4 engine build, both TensorRT-LLM curves if time allows, a vLLM/L4
curve regardless — needs a live GPU, so nothing in that list ran today.
What today produced instead: the L4-specific hardening the build script
needed anyway, and a mechanical version of the hard-stop decision itself,
so that when Colab does produce real build results, applying §6's rule
is "run a script and read its output," not a judgment call made while a
runtime is timing out.

**`build_trtllm_engine.py` hardened for the L4 case §21 specifically flags
as riskier.** Two changes:

1. **Card-aware output directory** (`--output-dir` now defaults to
   `/tmp/trtllm-engine-<card>` instead of a single hardcoded path) — building
   A100 then L4 back-to-back in one Colab session could otherwise have the
   second build's in-progress output silently land in the first's directory.
2. **`--build-workers`, defaulting to 1 on the L4 and unset (framework
   default) on the A100** (`_default_build_workers`). `trtllm-build
   --workers N` trades build parallelism for peak host RAM; §21 says the L4
   runtime "has 24 GB and less host RAM on most runtimes" than the A100
   one, so defaulting to a single worker there is a deliberate bet against
   an OOM'd build rather than discovering the tradeoff exists mid-build.
   Worth being explicit that this is a *build-time* host-RAM concern,
   unrelated to the *inference-time* KV-cache arithmetic already documented
   in `serving/config.py` (§18) — two different memory budgets, two
   different numbers, and conflating them would have been a real mistake to
   make quietly. 4 new tests confirm the card-aware default and that the
   `--workers` flag is threaded into the `trtllm-build` command only when
   set.

**`scripts/check_hard_stop.py`** — reads the build-log JSONL, takes the
latest FP16 record per card, and prints the exact verdict for one of four
outcomes (both engines serving / A100 only / L4 only / neither). While
writing this, noticed `SCOPE.md`'s original "Day 21 engine hard stop"
paragraph (written Day 1) was actually self-contradictory on close reading
— it said the *trigger* was "both engines not serving," but then described
reporting "the A100 crossover if that engine landed," which can't be true
under a trigger that requires both to have failed. **Rewrote that paragraph
today** into the explicit four-way rule the script now implements
verbatim, rather than leaving the ambiguity for whoever reads `SCOPE.md`
next under time pressure to resolve on the spot. This is exactly the kind
of small policy bug that's cheap to catch here, writing calmly on a CPU box
with no clock running, and expensive to catch live against a ticking Colab
session — which is the whole reason this script exists rather than trusting
the decision to memory. 10 tests, including one confirming a card that was
never attempted at all (not just "attempted and failed") is treated
identically to a failure rather than crashing on a missing key.

**Write:** `PAPER.md` §3.5 gained the portability-note paragraph (an
engine's SM-architecture binding is a sharper, more literal statement of
the hardware-dependence thesis than a configuration difference — "the
optimal binary does not exist on the other card at all") and a pointer from
Method to the hard-stop rule now living in both `SCOPE.md` and
`check_hard_stop.py`. §4.1's build-time table got its L4 row (still
pending data) with the `--workers 1` choice noted, plus a "Hard-stop
verdict" line to fill in once both rows exist. §5 Threats gained the
engine-build entry: build times and the `--workers` choice are
Colab-Pro-specific, not a portable benchmark of TensorRT-LLM's build cost
on other hardware.

**Build:** `scripts/build_trtllm_engine.py` (card-aware output dir,
`--build-workers`) · `scripts/check_hard_stop.py` · `SCOPE.md` (hard-stop
paragraph rewritten) · 14 new tests across `test_build_trtllm_engine.py`
and the new `test_check_hard_stop.py` — 88 total, all passing on CPU.

**Still queued for Colab (unchanged core items, carried forward):** the
TensorRT-LLM spike, vLLM concurrency-1 number, first real vLLM/A100 curve,
the actual A100 engine build + Drive verification, the real parity check —
and now the L4 engine build itself, run through `check_hard_stop.py` once
both cards' attempts exist. The hard-stop decision is data-dependent by
design and stays undetermined here; today's work is what makes that call
mechanical rather than improvised whenever it is actually made.

`[outcome: no GPU-dependent Day 21 deliverable executed (none can be, on
this box); build_trtllm_engine.py hardened for L4's tighter host RAM;
check_hard_stop.py written and tested against synthetic build-log fixtures,
and used to catch + fix a real self-contradiction in SCOPE.md's original
hard-stop wording. The actual hard-stop call is still queued for Colab]`

---

## Day 5 — Wed 5 Aug 2026 — **first GPU session**

First real hardware of the block. Colab Pro, A100-SXM4-40GB, driver 580.82.07.
Two things came out of it: a working vLLM install with a complete FP16
concurrency curve, and the discovery — from the server's own log, not from the
records — that the curve does not measure what it was supposed to measure.

### Install: the toolchain risk landed on the safe half

`M03-SERVING.md` §2.1 predicted the toolchain risk would be TensorRT-LLM. It
appeared on vLLM first, before TensorRT-LLM was touched at all.

`pip install vllm` succeeded but `import vllm` failed:

```
ImportError: libcudart.so.13: cannot open shared object file
```

**Not a driver problem.** `nvidia-smi` reported CUDA 13.0. The cause was inside
the pip environment: vLLM 0.26.0's PyPI wheel ships CUDA 13 binaries, while the
runtime's torch was `2.11.0+cu128`, a CUDA 12.8 build. The CUDA 13 runtime
*was* installed (`nvidia-cuda-runtime 13.3.29`, providing
`nvidia/cu13/lib/libcudart.so.13`) but that directory is not on the dynamic
loader's search path — torch's cu12 stack registers `nvidia/cuda_runtime/lib`
instead.

Fixed without reinstalling anything:

```bash
echo "/usr/local/lib/python3.12/dist-packages/nvidia/cu13/lib" \
  > /etc/ld.so.conf.d/cu13.conf && ldconfig
```

vLLM then imported and ran normally against the cu128 torch — it links through
the stable libtorch ABI (`vllm._C_stable_libtorch`), so the CUDA-version
mismatch between wheel and torch is not itself fatal; only the unresolvable
`.so` was.

Roughly 25 minutes lost, inside the 30-minute box. Two earlier attempts were
wasted re-running cells out of order rather than on the error itself — the
install is now a single ordered cell in the runbook.

**Threats entry this creates:** framework installation is CUDA-version-coupled,
and vLLM's default wheel assumed a newer CUDA runtime than the rented host's
torch provided. Install-time friction of this kind is absent from every
published framework comparison and belongs in a paper about framework choice.

### vLLM defaults observed at launch, and why they are recorded

From the startup banner, all confirmed against `config.py`: model
`Qwen/Qwen2.5-7B-Instruct`, revision `a09a3545…`, `max_model_len` 2048, dtype
float16. Three defaults were **not** specified by us and are now on the record:

| Default | Value | Why it matters |
|---|---|---|
| `enable_prefix_caching` | `True` | See below. This one invalidated the run's premise |
| `enable_chunked_prefill` | `True` | Changes how prefill and decode interleave under load |
| async scheduling | enabled | Affects the concurrency curve's shape |

Also logged: `WARNING Casting torch.bfloat16 to torch.float16`. Qwen2.5 is
natively bf16 and is being served in fp16 because `config.py` fixes fp16. Not
an error, but **the same cast must occur on TensorRT-LLM** or the two
frameworks are not serving identical numerics.

### The curve — clean, and measuring the wrong thing

24/24 records, every one `status: ok`, prompts_sha and model revision stamped.

| conc | mean tok/s | repeat spread | scaling eff. | TTFT p50 | TPOT p50 | client CPU |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 82.4 | 0.00% | 100% | 21.2 ms | 12.06 ms | 1.1% |
| 2 | 165.0 | 0.01% | 100.1% | 32.5 ms | 11.97 ms | 1.5% |
| 4 | 330.7 | 0.01% | 100.3% | 30.5 ms | 11.95 ms | 2.2% |
| 8 | 648.7 | 0.02% | 98.4% | 38.0 ms | 12.12 ms | 3.4% |
| 16 | 1257.2 | 1.66% | 95.3% | 41.1 ms | 12.42 ms | 5.3% |
| 32 | 2329.2 | 0.38% | 88.3% | 48.2 ms | 13.41 ms | 9.2% |
| 64 | 4098.9 | 1.51% | 77.7% | 63.7 ms | 15.27 ms | 15.3% |
| 128 | 6158.4 | 0.37% | 58.4% | 89.9 ms | 20.09 ms | 22.8% |

**Run-to-run spread is under 2% at every level, mostly under 0.5%.** Any
crossover worth reporting will sit far outside that band, which is the
condition §1's *Partial if* clause was written against. Client CPU peaked at
22.8% — the §4.2 harness gate holds under real server load, not only against
the mock.

**But the server log says the card was never loaded:**

```
Prefix cache hit rate:  96.9%
GPU KV cache usage:      5.2%     <- at concurrency 128
Running: 123 reqs, Waiting: 0 reqs
GPU KV cache size: 391,152 tokens
Maximum concurrency for 2,048 tokens per request: 190.99x
```

**Mechanism.** `data/prompts.json` holds **8** prompts. A 60-second window at
concurrency 128 issued ~2,944 requests, so each prompt ran ~368 times. After
its first use every repeat hit the prefix cache and skipped prefill outright.
At concurrency 128 the engine reported prompt throughput of 696 tok/s against
generation throughput of 6,346 tok/s — prefill had almost stopped happening.

Three consequences, all fatal to the run as a *main-sweep* result:

1. **TTFT is a cache-lookup time, not a prefill time.** 21 ms at concurrency 1
   is not what a 512-token prefill costs on this card.
2. **The A100 was never stressed.** 5.2% KV utilisation, zero queue depth, and
   capacity for 191 concurrent full-length requests against a load of 128. No
   crossover can appear anywhere in 1–128 by construction; the study would have
   measured nothing and looked like it had.
3. **It is an unmatched framework default.** TensorRT-LLM's prefix-reuse
   default is not vLLM's, so a Day 20 comparison against this curve would
   attribute a caching difference to the framework.

This was caught from the log rather than from the records, because
`kv_cache_util_pct`, `peak_vram_mb` and `gpu_util_pct` all came back **null** —
§8 requires them and the harness is not yet filling them. `kv_cache_util_pct`
is precisely the field that would have surfaced this automatically. Logged as
outstanding; not fixed today.

### Decision

**Prefix caching is OFF for the main sweep, on both frameworks**, set
explicitly in both directions rather than left to the default
(`scripts/serve_vllm.py`, `_PREFIX_CACHE_DEFAULT = False`). The launch command
now always states which regime a run was in.

Chosen over expanding the prompt set (which would break the frozen
`prompts_sha` and still leave caching as an unmatched default) and over
extending the concurrency axis (which would leave TTFT a cache-lookup number).

**The caching-ON curve is kept, not discarded** —
`results/vllm_a100_fp16_prefixcache_on.jsonl`. It is a legitimate measurement
of vLLM under high prompt reuse, and having both regimes is worth more than
having the intended one alone. Whether it earns a figure or a paragraph is a
Day 24 question.

`[outcome: partial — vLLM installs and serves on the A100; a complete,
low-variance 8-point FP16 curve exists but measures a cache-served workload
rather than a loaded card. Main-sweep curve to be re-run with prefix caching
disabled before any TensorRT-LLM comparison]`

**Next session:** re-run the A100 FP16 curve with caching off, confirm KV
utilisation and queue depth are non-trivial this time, then start the
TensorRT-LLM spike in a **separate runtime**.
