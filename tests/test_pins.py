"""Fast unit tests for pins.py's revision-pin lookup/storage. Same discipline
as Retrieval's test_pins.py (LOG.md, M01 Day 4 — three of four A100 models
had no recorded revision because the pin file lived only on a Colab VM that
got deleted). No network calls: `_resolve()` is monkeypatched wherever a pin
might need resolving, and `PINS_PATH` is redirected to a tmp_path file so the
real configs/model_pins.yaml is never touched.
"""

import pytest

from serving import pins


@pytest.fixture(autouse=True)
def isolated_pins(tmp_path, monkeypatch):
    monkeypatch.setattr(pins, "PINS_PATH", tmp_path / "model_pins.yaml")


def _forbid_network(monkeypatch):
    def _fail(model_id):
        raise AssertionError(f"_resolve() called for {model_id!r} — should have hit the pin file")

    monkeypatch.setattr(pins, "_resolve", _fail)


def test_revision_for_pinned_model_no_network_call(monkeypatch):
    pins.PINS_PATH.write_text("Qwen/Qwen2.5-7B-Instruct: abc123\n")
    _forbid_network(monkeypatch)
    assert pins.revision_for("Qwen/Qwen2.5-7B-Instruct") == "abc123"


def test_load_pins_never_rewrites_the_file():
    content = "Qwen/Qwen2.5-7B-Instruct: abc123\n"
    pins.PINS_PATH.write_text(content)
    before = pins.PINS_PATH.read_bytes()

    pins.load_pins()

    assert pins.PINS_PATH.read_bytes() == before


def test_revision_for_unpinned_id_resolves_once_then_reads_from_file(monkeypatch):
    calls = []
    monkeypatch.setattr(pins, "_resolve", lambda model_id: calls.append(model_id) or "resolved-sha")

    first = pins.revision_for("some/new-model")
    second = pins.revision_for("some/new-model")

    assert first == second == "resolved-sha"
    assert calls == ["some/new-model"]  # second call read the file, didn't re-resolve
    assert pins.load_pins()["some/new-model"] == "resolved-sha"


def test_all_pinned_revisions_resolves_each_id_once(monkeypatch):
    monkeypatch.setattr(pins, "_resolve", lambda model_id: f"sha-for-{model_id}")

    result = pins.all_pinned_revisions(["model/a", "model/b"])

    assert result == {"model/a": "sha-for-model/a", "model/b": "sha-for-model/b"}
