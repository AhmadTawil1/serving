"""Tests for the parts of build_trtllm_engine.py that don't need a GPU:
`_engine_id` (pure string logic) and `persist_to_drive` (plain filesystem
copy + verify -- exercised against tmp_path directories, not a real
multi-GB engine, but the verification logic itself doesn't care about file
content, only counts and sizes).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import build_trtllm_engine as bte  # noqa: E402


def test_engine_id_includes_card_and_precision():
    engine_id = bte._engine_id("A100-SXM4-40GB", "fp16")
    assert "A100-SXM4-40GB" in engine_id
    assert "fp16" in engine_id


def test_engine_id_is_filesystem_safe_no_slashes():
    engine_id = bte._engine_id("A100-SXM4-40GB", "fp16")
    assert "/" not in engine_id  # the model id has a "/" -- must not leak into a path segment


def _make_fake_engine(dir_: Path) -> None:
    dir_.mkdir(parents=True, exist_ok=True)
    (dir_ / "config.json").write_text('{"ok": true}')
    (dir_ / "rank0.engine").write_bytes(b"x" * 1000)


def test_persist_to_drive_succeeds_on_a_clean_copy(tmp_path):
    engine_dir = tmp_path / "engine"
    drive_dir = tmp_path / "drive" / "engine"
    _make_fake_engine(engine_dir)

    assert bte.persist_to_drive(engine_dir, drive_dir) is True
    assert (drive_dir / "config.json").read_text() == '{"ok": true}'
    assert (drive_dir / "rank0.engine").stat().st_size == 1000


def test_persist_to_drive_fails_verification_on_truncated_copy(tmp_path, monkeypatch):
    engine_dir = tmp_path / "engine"
    drive_dir = tmp_path / "drive" / "engine"
    _make_fake_engine(engine_dir)

    # Stub out the copy step and pre-seed drive_dir as if a copy already
    # happened but truncated one file -- isolates the verification logic
    # from the copy mechanism, and sidesteps monkeypatching shutil.copytree
    # with a replacement that would need to call the *original* copytree
    # (tricky: `bte.shutil` is the same module object anything else's
    # `import shutil` gets, so a naive replacement recurses into itself).
    monkeypatch.setattr(bte.shutil, "copytree", lambda *a, **k: None)
    drive_dir.mkdir(parents=True, exist_ok=True)
    (drive_dir / "config.json").write_text('{"ok": true}')
    (drive_dir / "rank0.engine").write_bytes(b"x" * 10)  # truncated vs. the 1000-byte original

    assert bte.persist_to_drive(engine_dir, drive_dir) is False


def test_persist_to_drive_fails_verification_on_missing_file(tmp_path, monkeypatch):
    engine_dir = tmp_path / "engine"
    drive_dir = tmp_path / "drive" / "engine"
    _make_fake_engine(engine_dir)

    monkeypatch.setattr(bte.shutil, "copytree", lambda *a, **k: None)
    drive_dir.mkdir(parents=True, exist_ok=True)
    (drive_dir / "config.json").write_text('{"ok": true}')
    # rank0.engine intentionally never created -- simulates a dropped file

    assert bte.persist_to_drive(engine_dir, drive_dir) is False
