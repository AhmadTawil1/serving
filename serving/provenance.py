"""Provenance stamping. REFERENCE.md ground rule 5: a result without
provenance is not a result -- every record's `prov` block comes from here.

Fields match the `prov` block in M03-SERVING.md §8's record schema exactly:
gpu, driver, cuda, torch, git_sha, model_revision, prompts_sha. Framework and
framework_version live in the record's `config` block instead (they vary per
run, not per environment) and are passed in by the caller, not computed here.

Never reimplement a hash to verify it (M01's `corpus_sha` false-alarm day,
carried forward via `REFERENCE.md` §7): `prompts_sha` is imported from the
module that writes it, never recomputed independently here.
"""

from __future__ import annotations

import subprocess

from . import pins


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "unknown"


def _nvidia_smi_field(field: str) -> str | None:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", f"--query-gpu={field}", "--format=csv,noheader"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
        return out.strip().splitlines()[0].strip()
    except Exception:
        return None


def stamp(model_id: str) -> dict:
    """The `prov` block for one result record.

    Stamped fresh at run time -- never typed from memory, never copied from a
    previous run. On the CPU dev box `gpu`/`driver` come back as the "no
    GPU" markers below; that is a correct provenance stamp for a structural
    smoke test here, not a placeholder to fill in later -- the real values
    are stamped fresh again when this same function runs on the rented card.
    """
    import torch

    from . import prompts

    if torch.cuda.is_available():
        gpu = _nvidia_smi_field("name") or torch.cuda.get_device_name(0)
        driver = _nvidia_smi_field("driver_version")
    else:
        gpu, driver = "none (CPU)", None

    return {
        "gpu": gpu,
        "driver": driver,
        "cuda": torch.version.cuda,
        "torch": torch.__version__,
        "git_sha": _git_sha(),
        "model_revision": pins.revision_for(model_id),
        "prompts_sha": prompts.prompts_sha(),
    }
