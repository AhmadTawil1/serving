"""Tests for sweep.py -- the concurrency-ladder driver."""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from aiohttp.test_utils import TestServer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import client as client_mod  # noqa: E402
import mock_server  # noqa: E402
import sweep  # noqa: E402
from serving import record as record_mod  # noqa: E402


def test_warmup_count_per_worker_distributes_total_across_workers():
    assert sweep._warmup_count_per_worker(32, 4) == 8
    assert sweep._warmup_count_per_worker(32, 128) == 1  # floors, but never below 1
    assert sweep._warmup_count_per_worker(1, 4) == 1


def test_run_sweep_writes_one_record_per_level_per_repeat(tmp_path, monkeypatch):
    monkeypatch.setattr(client_mod.config, "OUTPUT_TOKENS", 4)  # keep the test fast

    async def _t():
        server = TestServer(mock_server.create_app())
        await server.start_server()
        base_url = str(server.make_url(""))
        try:
            out_path = tmp_path / "out.jsonl"
            args = argparse.Namespace(
                base_url=base_url,
                framework="mock",
                framework_version=None,
                card="mock",
                precision="fp16",
                levels=[1, 2],
                window_s=0.1,
                repeats=2,
                warmup_requests=2,
                out=str(out_path),
            )
            await sweep.run_sweep(args)

            records = record_mod.read_records(out_path)
            assert len(records) == 4  # 2 levels x 2 repeats
            concurrencies = sorted(r["config"]["concurrency"] for r in records)
            assert concurrencies == [1, 1, 2, 2]
            repeats_seen = sorted({r["run"]["repeat"] for r in records})
            assert repeats_seen == [1, 2]
        finally:
            await server.close()

    asyncio.run(_t())


def test_run_sweep_stamps_provenance_and_config_on_every_record(tmp_path, monkeypatch):
    monkeypatch.setattr(client_mod.config, "OUTPUT_TOKENS", 4)

    async def _t():
        server = TestServer(mock_server.create_app())
        await server.start_server()
        base_url = str(server.make_url(""))
        try:
            out_path = tmp_path / "out.jsonl"
            args = argparse.Namespace(
                base_url=base_url,
                framework="mock",
                framework_version="v-test",
                card="mock",
                precision="int4",
                levels=[1],
                window_s=0.1,
                repeats=1,
                warmup_requests=1,
                out=str(out_path),
            )
            await sweep.run_sweep(args)

            [rec] = record_mod.read_records(out_path)
            assert rec["config"]["framework"] == "mock"
            assert rec["config"]["framework_version"] == "v-test"
            assert rec["config"]["precision"] == "int4"
            assert rec["prov"]["prompts_sha"]
            assert rec["prov"]["model_revision"]
        finally:
            await server.close()

    asyncio.run(_t())
