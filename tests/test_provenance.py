"""Provenance stamp tests. Isolated from the network (pins._resolve
monkeypatched) and from git/nvidia-smi (subprocess calls monkeypatched), so
this runs identically on the CPU dev box and on a rented GPU host.
"""

from __future__ import annotations

import pytest

from serving import pins, prompts, provenance


@pytest.fixture(autouse=True)
def isolated_pins(tmp_path, monkeypatch):
    monkeypatch.setattr(pins, "PINS_PATH", tmp_path / "model_pins.yaml")


@pytest.fixture(autouse=True)
def isolated_prompts(tmp_path, monkeypatch):
    frozen = tmp_path / "prompts.json"
    frozen.write_text('{"prompts": ["hello"]}')
    monkeypatch.setattr(prompts, "PROMPTS_PATH", frozen)


def test_stamp_has_every_schema_field(monkeypatch):
    monkeypatch.setattr(pins, "_resolve", lambda model_id: "resolved-sha")
    monkeypatch.setattr(provenance, "_git_sha", lambda: "test-git-sha")
    monkeypatch.setattr(provenance, "_nvidia_smi_field", lambda field: None)

    prov = provenance.stamp("some/model")

    # Matches M03-SERVING.md §8's record schema `prov` block exactly.
    assert set(prov) == {
        "gpu",
        "driver",
        "cuda",
        "torch",
        "git_sha",
        "model_revision",
        "prompts_sha",
    }
    assert prov["git_sha"] == "test-git-sha"
    assert prov["model_revision"] == "resolved-sha"


def test_stamp_reports_cpu_when_no_cuda(monkeypatch):
    import torch

    monkeypatch.setattr(pins, "_resolve", lambda model_id: "sha")
    monkeypatch.setattr(provenance, "_git_sha", lambda: "sha")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    prov = provenance.stamp("some/model")

    assert prov["gpu"] == "none (CPU)"
    assert prov["driver"] is None


def test_stamp_prompts_sha_matches_prompts_module(monkeypatch):
    monkeypatch.setattr(pins, "_resolve", lambda model_id: "sha")
    monkeypatch.setattr(provenance, "_git_sha", lambda: "sha")

    prov = provenance.stamp("some/model")

    assert prov["prompts_sha"] == prompts.prompts_sha()
