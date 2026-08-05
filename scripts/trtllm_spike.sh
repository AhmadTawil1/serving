#!/usr/bin/env bash
# TensorRT-LLM install spike (M03-SERVING.md §6, §9). Timeboxed to 30
# minutes. The question this answers is NOT "does the study model build" --
# it is "does this toolchain install and build *anything* on this runtime at
# all." Smallest available model, not the study model.
#
# Linux + NVIDIA GPU only. Confirmed on the Windows dev box (5 Aug 2026,
# `uv pip install --dry-run tensorrt-llm`) that no Windows wheel exists at
# any version -- PyPI only publishes `linux_x86_64` / `linux_aarch64` for
# this package, and the plain-PyPI `tensorrt-llm` entry is a stub that
# redirects to https://pypi.nvidia.com/ besides. So this script is written
# here and RUN on Colab, against the A100 runtime, not on the dev box.
#
# Usage, in a Colab cell:
#   !bash scripts/trtllm_spike.sh 2>&1 | tee data/trtllm_spike_a100.log
set -euo pipefail

SECONDS=0
BUDGET_S=$((30 * 60))
LOG_STEP() { echo "[trtllm_spike] +${SECONDS}s: $*"; }

check_budget() {
    if (( SECONDS > BUDGET_S )); then
        LOG_STEP "OVER BUDGET (${SECONDS}s > ${BUDGET_S}s) -- stopping. This is the Day 18 answer: install/build alone exceeds the 30-minute spike box on this runtime. Record it verbatim in LOG.md; do not let this become a full build attempt."
        exit 1
    fi
}

LOG_STEP "nvidia-smi:"
nvidia-smi || { LOG_STEP "no GPU visible -- wrong runtime, stop here"; exit 1; }

LOG_STEP "installing tensorrt-llm from the NVIDIA index"
pip install --extra-index-url https://pypi.nvidia.com/ tensorrt-llm
check_budget

LOG_STEP "tensorrt-llm import + version check"
python -c "import tensorrt_llm; print(tensorrt_llm.__version__)"
check_budget

# Smallest readily-available checkpoint for a fast install/build smoke test
# -- NOT the study model (Qwen2.5-7B-Instruct). The point is toolchain
# function, not a real number.
SPIKE_MODEL="gpt2"

LOG_STEP "cloning TensorRT-LLM examples for the smallest supported model family"
git clone --depth 1 https://github.com/NVIDIA/TensorRT-LLM.git /tmp/trtllm-examples
check_budget

LOG_STEP "converting ${SPIKE_MODEL} checkpoint to TensorRT-LLM format"
python /tmp/trtllm-examples/examples/gpt/convert_checkpoint.py \
    --model_dir "${SPIKE_MODEL}" \
    --output_dir /tmp/trtllm-ckpt \
    --dtype float16
check_budget

LOG_STEP "building the engine"
trtllm-build --checkpoint_dir /tmp/trtllm-ckpt --output_dir /tmp/trtllm-engine
check_budget

LOG_STEP "SUCCESS -- toolchain installs and builds on this runtime in ${SECONDS}s. Record this wall time in LOG.md; it is the Day 18 number's second half."
