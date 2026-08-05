"""Model revision pinning, shared by every model the harness serves.

Ground rule (REFERENCE.md §7; carried over from Retrieval's and Ingestion's
`pins.py`): every model reference goes through `revision_for()`, never a bare
model id, and every pinned revision is stamped into each record's `prov`
block rather than left in a file that may never be committed back from a
deleted Colab VM.

Keys are the Hugging Face model IDs themselves, so a pin is self-describing
and cannot drift from the model it pins.
"""

from __future__ import annotations

from pathlib import Path

import yaml

PINS_PATH = Path("configs/model_pins.yaml")


def load_pins() -> dict[str, str]:
    if PINS_PATH.exists():
        return yaml.safe_load(PINS_PATH.read_text()) or {}
    return {}


def _save_pin(key: str, revision: str) -> None:
    pins = load_pins()
    pins[key] = revision
    PINS_PATH.parent.mkdir(parents=True, exist_ok=True)
    PINS_PATH.write_text(yaml.safe_dump(pins, sort_keys=True))


def _resolve(model_id: str) -> str:
    from huggingface_hub import HfApi

    return HfApi().model_info(model_id).sha


def revision_for(model_id: str) -> str:
    """The frozen revision for `model_id`, resolving and pinning on first use."""
    pins = load_pins()
    if model_id in pins:
        return pins[model_id]

    revision = _resolve(model_id)
    _save_pin(model_id, revision)
    return revision


def all_pinned_revisions(model_ids: list[str]) -> dict[str, str]:
    """{model_id: revision} for every model the harness uses -- goes into `prov`."""
    return {mid: revision_for(mid) for mid in model_ids}
