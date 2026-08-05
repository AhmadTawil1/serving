"""Frozen study configuration, in one place so every script imports the same
constants rather than typing them twice (M03-SERVING.md §3 "held fixed").

Every value here is a Day 18 decision, recorded with its reasoning in
`LOG.md`. Changing anything in this file after Day 18 is a scope change and
belongs in `SCOPE.md`, not a silent edit.
"""

from __future__ import annotations

# --- Model (§18 "the M03 equivalent of M02's embedder choice") -------------
#
# Qwen2.5-7B-Instruct, chosen over an 8B Llama-class model for two reasons:
#  1. GQA with 4 KV heads (vs. Llama-3.1-8B's 8) roughly halves the per-token
#     KV-cache footprint, which is what makes concurrency 128 fit on a 24 GB
#     L4 at all -- see the arithmetic below.
#  2. Ungated on the Hub. Llama-3.1-8B-Instruct requires per-user license
#     acceptance and an HF_TOKEN; a gated model adds a reproducibility
#     dependency (a token, tied to one account) this study doesn't need.
STUDY_MODEL = "Qwen/Qwen2.5-7B-Instruct"

# --- Sequence lengths (§18 "pick one representative pair, fix it") ---------
#
# Matches the worked example in M03-SERVING.md §8's record schema. One
# representative prefill/decode balance, fixed and reported as a scope limit
# rather than swept.
INPUT_TOKENS = 512
OUTPUT_TOKENS = 128

# Max model length must be identical across frameworks and cards -- it
# changes achievable KV-cache concurrency (§3 "held fixed"). Set well above
# input+output (640) so no request is truncated, but not so large it wastes
# reserved KV-cache blocks.
MAX_MODEL_LEN = 2048

# --- SLO thresholds for goodput (§18, "working assumption") ----------------
SLO_TTFT_MS = 1000
SLO_TPOT_MS = 50

# --- Sweep axes (§3, frozen 18 Aug) -----------------------------------------
CONCURRENCY_LEVELS = [1, 2, 4, 8, 16, 32, 64, 128]
PRECISIONS = ["fp16", "fp8", "int8", "int4"]
CARDS = ["A100-SXM4-40GB", "L4-24GB"]
FRAMEWORKS = ["vllm", "tensorrt-llm"]

# --- L4 24 GB arithmetic (§18, done explicitly rather than assumed) --------
#
# Qwen2.5-7B-Instruct config (fetched from the Hub, not guessed):
#   hidden_size=3584, num_hidden_layers=28, num_key_value_heads=4,
#   head_dim=128, native dtype bf16.
#
# Weights (bf16, 2 bytes/param, ~7.61B params):
#   7.61e9 * 2 bytes ~= 14.2 GiB
#
# KV cache per token (2 for K+V, bf16):
#   2 * 28 layers * 4 kv_heads * 128 head_dim * 2 bytes = 57,344 bytes
#   = 56 KiB/token
#
# KV cache per request at INPUT_TOKENS + OUTPUT_TOKENS = 640 tokens:
#   640 * 56 KiB ~= 35.0 MiB
#
# KV cache at concurrency 128 (every worker holding one full-length request,
# the closed-loop worst case, §4.2):
#   128 * 35.0 MiB ~= 4.4 GiB
#
# Weights + KV cache at concurrency 128: 14.2 + 4.4 = 18.6 GiB, against a
# 24 GiB card. That leaves ~5.4 GiB for activations, CUDA context, and
# framework overhead -- vLLM's own `gpu_memory_utilization` knob (commonly
# 0.85-0.90 of the card) effectively budgets ~20.4-21.6 GiB, so the 18.6 GiB
# figure fits with a ~2-3 GiB margin. Tight but workable; the honest read is
# that concurrency 128 on the L4 is plausible, not comfortable, and an OOM
# there is itself a result per §17 ("L4 OOMs above concurrency 32 -- that is
# the finding"), not a harness bug to chase.
#
# On the A100 (40 GiB) the same arithmetic leaves ~21.4 GiB of headroom --
# not the constraint on that card.
