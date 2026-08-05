"""Engine-build timing/resource logging.

§6: "Engine build time is a reported metric, not overhead" -- a framework
that serves faster but costs three hours per card per precision to build
has a real operational cost nobody publishes, so wall time, peak host RAM
and peak VRAM are captured for *every* build attempt, including failed
ones (§6: "record every failed attempt with its error, not just the
successful one" -- the failure list is what makes the eventual build-time
table in §4 of `PAPER.md` credible).

Generic over the command run -- `scripts/build_trtllm_engine.py` calls
`run_timed_with_peaks` once per build step (checkpoint conversion,
`trtllm-build`, ...) and folds the per-step results into one `build_record`.
"""

from __future__ import annotations

import json
import subprocess
import threading
import time
from pathlib import Path

import psutil


def peak_host_ram_mb_sampler(
    samples: list[float], stop_event: threading.Event, interval_s: float = 1.0
) -> None:
    """Runs in a background thread for the duration of a build step,
    appending host RAM (used, MB) samples. Whole-system, not just this
    process's RSS -- a build spawns its own subprocesses (checkpoint
    conversion, `trtllm-build`), so per-process RSS would undercount.
    """
    while not stop_event.is_set():
        samples.append(psutil.virtual_memory().used / (1024 * 1024))
        stop_event.wait(interval_s)


def _nvidia_smi_used_mb() -> float | None:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=5,
        )
        return float(out.strip().splitlines()[0])
    except Exception:
        return None


def peak_vram_mb_sampler(
    samples: list[float], stop_event: threading.Event, interval_s: float = 1.0
) -> None:
    """Same idea for GPU VRAM, via `nvidia-smi`. Appends nothing on a host
    with no GPU (the dev box), so the caller sees an empty list and
    `peak_vram_mb` comes back `None` -- the same "none (CPU)" spirit as
    `serving/provenance.py`'s `stamp()`.
    """
    while not stop_event.is_set():
        used = _nvidia_smi_used_mb()
        if used is not None:
            samples.append(used)
        stop_event.wait(interval_s)


def run_timed_with_peaks(cmd: list[str], **subprocess_kwargs) -> dict:
    """Run `cmd` to completion, timing it and sampling peak host RAM and
    peak VRAM throughout. Captures stdout/stderr regardless of outcome, so a
    failed step's exact error is on hand to record (§6), not just its
    return code.
    """
    ram_samples: list[float] = []
    vram_samples: list[float] = []
    stop_event = threading.Event()
    ram_thread = threading.Thread(
        target=peak_host_ram_mb_sampler, args=(ram_samples, stop_event)
    )
    vram_thread = threading.Thread(
        target=peak_vram_mb_sampler, args=(vram_samples, stop_event)
    )
    ram_thread.start()
    vram_thread.start()

    t0 = time.monotonic()
    proc = subprocess.run(cmd, capture_output=True, text=True, **subprocess_kwargs)
    wall_s = time.monotonic() - t0

    stop_event.set()
    ram_thread.join(timeout=5)
    vram_thread.join(timeout=5)

    return {
        "cmd": cmd,
        "wall_s": wall_s,
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
        "peak_host_ram_mb": max(ram_samples) if ram_samples else None,
        "peak_vram_mb": max(vram_samples) if vram_samples else None,
    }


def build_record(
    *,
    card: str,
    precision: str,
    framework: str,
    engine_id: str | None,
    step_results: list[dict],
    succeeded: bool,
) -> dict:
    """One build-log entry -- §4's build-time table, one row per attempt.
    `step_results` is a list of `run_timed_with_peaks` outputs, one per
    build step; this folds them into totals without discarding the
    per-step detail (kept under `"steps"` so a failed step's stderr is
    still reachable).
    """
    host_peaks = [s["peak_host_ram_mb"] for s in step_results if s["peak_host_ram_mb"] is not None]
    vram_peaks = [s["peak_vram_mb"] for s in step_results if s["peak_vram_mb"] is not None]
    return {
        "card": card,
        "precision": precision,
        "framework": framework,
        "engine_id": engine_id,
        "succeeded": succeeded,
        "total_wall_s": sum(s["wall_s"] for s in step_results),
        "peak_host_ram_mb": max(host_peaks) if host_peaks else None,
        "peak_vram_mb": max(vram_peaks) if vram_peaks else None,
        "steps": step_results,
    }


def write_build_record(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(record) + "\n")
