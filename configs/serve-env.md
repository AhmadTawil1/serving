# GPU-host environment

This project's `pyproject.toml` deliberately does **not** install vLLM or
TensorRT-LLM (see the comment there). Both are Linux + CUDA only, both pin
their own `torch`/`transformers` versions that conflict with what this
project's own tooling needs, and neither has a Windows wheel at all
(confirmed 5 Aug 2026 — see `LOG.md` Day 1). They are installed in their own
environment on the rented GPU host (Colab, A100/L4), separate from the `uv`
environment this repo's own code runs in.

## On Colab

```bash
git clone <this repo> && cd Serving
pip install vllm                                   # its own env; do not `uv sync` here
python scripts/serve_vllm.py --card A100-SXM4-40GB --precision fp16
```

`scripts/serve_vllm.py` imports `serving.config`/`serving.pins`/`serving.prompts`
directly (plain `sys.path` insert, no package install needed) so it does not
require this project's own `uv` environment either — only `vllm`, `requests`,
`pyyaml`, and `huggingface_hub` need to be importable, all of which vLLM's own
install pulls in except `pyyaml`/`requests` (`pip install pyyaml requests`
alongside vLLM if they're not already present).

## TensorRT-LLM

Same story, plus the Day 18 spike (`scripts/trtllm_spike.sh`) has to run
there too — see that file's header for why it cannot run on the dev box at
all (no Windows wheel published, at any version).

## Dev box (this machine)

CPU-only, Windows. `uv sync` here installs everything *except* vLLM/TRT-LLM:
the `serving` package (`pins.py`, `provenance.py`, `prompts.py`, `quality.py`,
`config.py`), and the test suite, all of which are structurally validated
without a GPU. See `M03-SERVING.md` §5 and `REFERENCE.md` — this mirrors how
`Retrieval` and `Ingestion` were built (CPU smoke tests here, real numbers on
the rented card).
