"""Moving a training job from a slower GPU to a faster one when it frees up (server.JobManager, 1.7.8): the slower
slot starts a waiting job (block jobs too) while the faster one is busy; when the faster one frees, the job pauses
(its state saved), goes back to the queue pinned to the faster slot and continues there from the same step, under
the same jobId. Fake slots and a fake pausable trainer."""
from __future__ import annotations

import json
import sys
import threading
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

pytest.importorskip("fastapi")

from splatworker.backends import JobCancelled, JobPaused  # noqa: E402
from splatworker.slots import Slot  # noqa: E402

TOTAL = 60


class FakeVulkan:
    """Steps until ``total``; pauses on request saving the step in state.txt, resumes from it; the slower device
    (Vulkan index 1) takes longer per step. Reports ETA × ``eta_scale`` so migration looks worth it."""
    pausable = True
    leases = False

    def __init__(self):
        self.runs = []                   # (job name, vulkan index, start step)
        self.hold = {}                   # name → Event: the job holds at its first step until set
        self.delay = {0: 0.005, 1: 0.03}
        self.eta_scale = 100.0

    def capabilities(self):
        return ["train"]

    def census_available(self):
        return False

    def trainer_choice(self):
        return {"trainer": "vulkan", "reason": "test"}

    def models(self):
        return []

    def run(self, ctx):
        name = ctx.spec.get("name")
        total = int(ctx.spec.get("iterations", TOTAL))
        state = ctx.job_dir / "state.txt"
        start = int(state.read_text()) if state.is_file() else 0
        self.runs.append((name, ctx.vulkan_index, start))
        hold = self.hold.get(name)
        for step in range(start, total):
            if hold is not None and not hold.is_set():
                while not hold.wait(0.02):
                    if ctx.cancel_event.is_set():
                        raise JobCancelled()
            if ctx.cancel_event.is_set():
                raise JobCancelled()
            if ctx.pause_event.is_set():
                state.write_text(str(step))
                raise JobPaused(step)
            time.sleep(self.delay.get(ctx.vulkan_index, 0.01))
            ctx.status(state="training", step=step + 1, totalSteps=total,
                       etaSeconds=(total - step - 1) * self.delay.get(ctx.vulkan_index, 0.01) * self.eta_scale)
        state.unlink(missing_ok=True)
        (ctx.job_dir / "result.ply").write_bytes(b"ply\n")
        return {"numGaussians": 1}


def wait_for(cond, timeout=15.0):
    end = time.time() + timeout
    while time.time() < end:
        v = cond()
        if v:
            return v
        time.sleep(0.01)
    raise AssertionError("timed out")


FAST = dict(key=0, cuda=0, vulkan=0, name="RTX 4080", memory_mb=16376)
SLOW = dict(key="vk:0000:11:00.0", vulkan=1, name="Radeon 610M", integrated=True, slow=True, slower=True,
            memory_mb=2048)


@pytest.fixture
def worker(tmp_path):
    from splatworker.config import ServerSettings
    from splatworker.server import JobManager

    def make(migrate=True, slow_choice="auto"):
        settings = ServerSettings(data_dir=tmp_path / "data", mdns=False, migrate=migrate)
        for d in (settings.datasets_dir, settings.jobs_dir):
            d.mkdir(parents=True, exist_ok=True)
        ds = settings.datasets_dir / "d1"
        (ds / "files").mkdir(parents=True, exist_ok=True)
        (ds / "meta.json").write_text("{}")
        slow = Slot(**{**SLOW, "slow": slow_choice != "always", "user_choice": slow_choice})
        backend = FakeVulkan()
        m = JobManager(settings, backend, [Slot(**FAST), slow])
        m.start()
        made.append(m)
        return m, backend
    made = []
    yield make
    for m in made:
        for e in m.backend.hold.values():
            e.set()
        m.shutdown(timeout=5)


def submit(m, name, block=False, iterations=TOTAL):
    cfg = {"name": name, "iterations": iterations, "maxGaussians": 100_000}
    if block:
        cfg["region"] = {"min": [0, 0, 0], "max": [1, 1, 1]}
    return m.submit("train", "d1", name, "config", cfg, total_steps=iterations)


def test_block_starts_on_the_slower_gpu_and_moves_to_the_faster_one(worker):
    m, b = worker()
    b.hold["A"] = threading.Event()
    a = submit(m, "A", block=True)
    wait_for(lambda: ("A", 0, 0) in b.runs)                    # block 1 on the 4080 (held)
    bj = submit(m, "B", block=True)
    wait_for(lambda: ("B", 1, 0) in b.runs)                    # block 2 starts on the 610M meanwhile
    wait_for(lambda: (bj.status.get("step") or 0) >= 5)
    assert bj.status["slot"] == "vk:0000:11:00.0" and bj.status["gpu"] == "Radeon 610M"
    seen = []
    b.hold["A"].set()                                          # block 1 finishes: the 4080 frees up
    wait_for(lambda: (seen.append(bj.status.get("step") or 0) or len([r for r in b.runs if r[0] == "B"]) == 2))
    moved_at = [r for r in b.runs if r[0] == "B"][1]
    assert moved_at[1] == 0 and moved_at[2] > 0                # resumed on Vulkan device 0 from its step
    wait_for(lambda: bj.state == "completed" and a.state == "completed")
    assert bj.status["step"] == TOTAL and bj.status["slot"] == "cuda:0" and bj.status["gpu"] == "RTX 4080"
    assert all(y >= x for x, y in zip(seen, seen[1:]))          # the step never went back
    log = (bj.dir / "log.txt").read_text() if (bj.dir / "log.txt").exists() else "".join(
        p.read_text() for p in bj.dir.glob("*.log"))
    assert f"moved from Radeon 610M to RTX 4080 at step {moved_at[2]}" in log
    assert not m.reserved and "pin" not in bj.spec


def test_no_move_when_almost_done_or_turned_off(worker):
    m, b = worker()
    b.eta_scale = 0.0                                          # it reports almost no time left
    b.hold["A"] = threading.Event()
    submit(m, "A")
    wait_for(lambda: ("A", 0, 0) in b.runs)
    bj = submit(m, "B")
    wait_for(lambda: (bj.status.get("step") or 0) >= 3)
    b.hold["A"].set()
    wait_for(lambda: bj.state == "completed")
    assert [r for r in b.runs if r[0] == "B"] == [("B", 1, 0)]  # finished where it was

    m2, b2 = worker(migrate=False)
    b2.hold["A"] = threading.Event()
    submit(m2, "A")
    wait_for(lambda: ("A", 0, 0) in b2.runs)
    cj = submit(m2, "B")
    wait_for(lambda: (cj.status.get("step") or 0) >= 3)
    b2.hold["A"].set()
    wait_for(lambda: cj.state == "completed")
    assert [r for r in b2.runs if r[0] == "B"] == [("B", 1, 0)]


def test_cancel_and_pause_while_moving(worker):
    m, b = worker()
    b.delay[1] = 0.2                                           # the 610M is slow: time to act mid-move
    b.hold["A"] = threading.Event()
    submit(m, "A")
    wait_for(lambda: ("A", 0, 0) in b.runs)
    bj = submit(m, "B")
    wait_for(lambda: (bj.status.get("step") or 0) >= 1)
    b.hold["A"].set()
    wait_for(lambda: bj.migrate_to is not None or m.reserved)
    m.cancel(bj.id)
    wait_for(lambda: bj.state == "cancelled")
    wait_for(lambda: not m.reserved)
    c = submit(m, "C")                                          # the 4080 isn't held for the cancelled job
    wait_for(lambda: c.state == "completed")

    m2, b2 = worker()
    b2.delay[1] = 0.2
    b2.hold["A"] = threading.Event()
    submit(m2, "A")
    wait_for(lambda: ("A", 0, 0) in b2.runs)
    dj = submit(m2, "D")
    wait_for(lambda: (dj.status.get("step") or 0) >= 1)
    b2.hold["A"].set()
    wait_for(lambda: dj.migrate_to is not None)
    m2.pause(dj.id)                                             # the user's pause wins: it stays paused
    wait_for(lambda: dj.state == "paused")
    assert not m2.reserved and "pin" not in dj.spec
    m2.resume(dj.id)
    wait_for(lambda: dj.state == "completed")
    assert dj.status["step"] == TOTAL


def test_always_use_runs_both_gpus_in_parallel(worker):
    m, b = worker(migrate=False, slow_choice="always")
    b.hold["A"] = threading.Event()
    b.hold["B"] = threading.Event()
    submit(m, "A", block=True)
    submit(m, "B", block=True)
    wait_for(lambda: {r[1] for r in b.runs} == {0, 1})          # one block on each GPU at once
    b.hold["A"].set()
    b.hold["B"].set()


def test_a_move_interrupted_by_a_restart(tmp_path):
    from splatworker.config import ServerSettings
    from splatworker.server import JobManager

    settings = ServerSettings(data_dir=tmp_path / "data", mdns=False)
    jd = settings.jobs_dir / "jabc"
    jd.mkdir(parents=True)
    (jd / "job.json").write_text(json.dumps({
        "spec": {"kind": "train", "datasetId": "d1", "name": "B", "config": {}, "pin": "cuda:0",
                 "movedFrom": "Radeon 610M"},
        "status": {"jobId": "jabc", "kind": "train", "state": "queued", "step": 30, "createdAt": "x"}}))
    m = JobManager(settings, FakeVulkan(), [Slot(**FAST), Slot(**SLOW)])
    m.load()
    job = m.jobs["jabc"]
    assert "pin" not in job.spec and "movedFrom" not in job.spec and job.state == "queued"
