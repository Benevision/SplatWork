"""GPU leases (splatworker/gpulease.py): processing jobs run in a CPU pool and hold their GPU only around GPU
stages, so they do their CPU work while training runs; a GPU stage waits for the GPU, or runs beside training when
its free memory allows. Fake stages, no GPU."""
from __future__ import annotations

import json
import queue
import sys
import threading
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from splatworker import gpulease  # noqa: E402

SFM = ["pairs", "features", "matching", "verification", "mapping", "alignment", "export"]


def test_which_stages_need_the_gpu():
    assert [s for s in SFM if gpulease.needs_gpu("sfm", s)] == ["pairs", "features", "matching"]
    assert gpulease.needs_gpu("masks", "masks") and gpulease.needs_gpu("maskpreview", "anything")
    assert not gpulease.needs_gpu("frames", "extracting frames")             # plain video decoding: CPU
    assert not gpulease.needs_gpu("frames", "restoring home video")         # VHS: CPU
    assert gpulease.needs_gpu("frames", "splitting 360° video")             # resampling on torch
    assert gpulease.needs_gpu("newtask", "x")                               # unknown: the GPU throughout
    assert not gpulease.needs_gpu("masks", "masks", {"gpu": False})          # splatprep may say so itself
    assert gpulease.need_mb("sfm", "matching") == 4000 and gpulease.need_mb("masks", "masks") == 6000


def test_gate_acquires_once_and_releases_for_cpu_stages():
    q, granted, statuses = queue.Queue(), threading.Event(), []
    gate = gpulease.Gate(q, granted, lambda: False, lambda **f: statuses.append(f))
    granter = threading.Thread(target=lambda: (q.get(timeout=5), granted.set()))
    granter.start()
    for st in SFM[:3]:
        gate.stage("sfm", st)
    granter.join()
    assert gate.holding and q.empty()                            # one acquire for pairs + features + matching
    gate.stage("sfm", "verification")
    assert q.get_nowait() == ("gpu", {"release": True}) and not granted.is_set() and not gate.holding
    for st in SFM[4:]:
        gate.stage("sfm", st)
    assert q.empty()
    assert [s["resource"] for s in statuses] == ["waiting-gpu", "gpu", "cpu"]
    assert statuses[0]["message"] == "pairs: waiting for the GPU"


def test_gate_cancelled_while_waiting():
    q, granted = queue.Queue(), threading.Event()
    stop = threading.Event()
    gate = gpulease.Gate(q, granted, stop.is_set, lambda **f: None)
    threading.Timer(0.3, stop.set).start()
    with pytest.raises(KeyboardInterrupt):
        gate.stage("masks", "masks")


# ------------------------------------------------------------- the worker

class LeaseBackend:
    """Train jobs run until ``release_train`` is set; processing jobs do CPU work, then one GPU stage (via the
    lease), then more CPU work, each step waiting for its event so the test can look in between."""
    pausable = False
    leases = True

    def __init__(self):
        self.release_train = threading.Event()
        self.events = {}
        self.log = []
        self.lock = threading.Lock()

    def ev(self, name):
        with self.lock:
            return self.events.setdefault(name, threading.Event())

    def capabilities(self):
        return ["train", "process"]

    def census_available(self):
        return False

    def trainer_choice(self):
        return {"trainer": "gsplat", "reason": "test"}

    def models(self):
        return []

    def run(self, ctx):
        name = ctx.spec.get("name", ctx.job_id)
        if ctx.kind == "train":
            ctx.status(state="training", step=1)
            self.log.append(("train-start", name))
            self.release_train.wait(20)
            self.log.append(("train-end", name))
            (ctx.job_dir / "result.ply").write_bytes(b"ply\n")
            return {}
        assert ctx.gpu_request is not None
        ctx.status(state="running", resource="cpu", message="pairs")
        self.log.append(("cpu", name))
        self.ev(f"{name}:cpu1").wait(20)
        granted = threading.Event()
        ctx.status(resource="waiting-gpu", message="features: waiting for the GPU")
        ctx.gpu_request("features", 3000, granted.set)
        assert granted.wait(20), "no GPU lease"
        ctx.status(resource="gpu", message="features")
        self.log.append(("gpu", name))
        self.ev(f"{name}:gpu").wait(20)
        ctx.gpu_release()
        ctx.status(resource="cpu", message="mapping")
        self.log.append(("mapping", name))
        self.ev(f"{name}:cpu2").wait(20)
        out = ctx.job_dir / "output"
        out.mkdir(exist_ok=True)
        (out / "x.txt").write_text("x")
        return {}


def wait_for(cond, timeout=10.0):
    end = time.time() + timeout
    while time.time() < end:
        v = cond()
        if v:
            return v
        time.sleep(0.02)
    raise AssertionError("timed out")


@pytest.fixture
def worker(tmp_path):
    pytest.importorskip("fastapi")
    from splatworker.config import ServerSettings
    from splatworker.server import JobManager

    settings = ServerSettings(data_dir=tmp_path, mdns=False, cpu_jobs=1)
    for d in (settings.datasets_dir, settings.jobs_dir):
        d.mkdir(parents=True, exist_ok=True)
    ds = settings.datasets_dir / "d1"
    (ds / "files").mkdir(parents=True)
    (ds / "meta.json").write_text("{}")
    backend = LeaseBackend()
    m = JobManager(settings, backend, [0])
    m.free = 20_000
    m._free_mb = lambda slot: m.free
    m.start()
    yield m, backend
    backend.release_train.set()
    for e in list(backend.events.values()):
        e.set()
    m.shutdown(timeout=5)


def submit(m, kind, name, task="sfm"):
    if kind == "train":
        return m.submit("train", "d1", name, "config", {"name": name}, total_steps=10)
    return m.submit("process", "d1", name, "request", {"name": name, "task": task}, total_steps=1000)


def test_processing_does_its_cpu_work_while_training_runs(worker):
    m, b = worker
    t = submit(m, "train", "T")
    wait_for(lambda: ("train-start", "T") in b.log)
    p = submit(m, "process", "P")
    wait_for(lambda: ("cpu", "P") in b.log)                       # not waiting behind the training job
    assert p.status["resource"] == "cpu" and p.state in ("loading", "running", "training")
    m.free = 2000                                                 # training has the GPU's memory: the stage waits
    b.ev("P:cpu1").set()
    wait_for(lambda: p.status.get("resource") == "waiting-gpu")
    time.sleep(0.5)
    assert ("gpu", "P") not in b.log and [x.job_id for x in m.pending] == [p.id]
    m.free = 12_000                                               # enough for features beside training
    wait_for(lambda: ("gpu", "P") in b.log, timeout=6)
    assert m.leases == {0: p.id} and m.running == {0: t.id}       # both on GPU 0 at once
    assert "GPU granted" in (p.dir / "log.txt").read_text() if (p.dir / "log.txt").exists() else True
    b.ev("P:gpu").set()
    wait_for(lambda: ("mapping", "P") in b.log)
    wait_for(lambda: not m.leases)
    b.ev("P:cpu2").set()
    wait_for(lambda: p.state == "completed")
    assert t.state not in ("completed", "failed")                 # training kept going throughout
    b.release_train.set()
    wait_for(lambda: t.state == "completed")


def test_training_first_and_the_gpu_held_by_a_stage(worker):
    m, b = worker
    b.ev("P:cpu1").set()
    p = submit(m, "process", "P")
    wait_for(lambda: ("gpu", "P") in b.log)                       # a free GPU: granted at once
    t = submit(m, "train", "T")
    time.sleep(1.5)
    assert ("train-start", "T") not in b.log                      # the GPU stage has it; training waits
    b.ev("P:gpu").set()
    wait_for(lambda: ("train-start", "T") in b.log)               # handed back → training starts
    # now a second processing job's GPU stage: training holds the GPU, too little memory → waits
    m.free = 1000
    b.ev("Q:cpu1").set()
    q = submit(m, "process", "Q")
    b.ev("P:cpu2").set()
    wait_for(lambda: q.status.get("resource") == "waiting-gpu")
    b.release_train.set()
    wait_for(lambda: ("gpu", "Q") in b.log)                       # training finished → the stage gets the GPU
    for e in ("Q:gpu", "Q:cpu2"):
        b.ev(e).set()
    wait_for(lambda: q.state == "completed" and p.state == "completed" and t.state == "completed")


def test_cpu_limit_and_the_quick_lane(worker):
    m, b = worker
    a = submit(m, "process", "A")
    wait_for(lambda: ("cpu", "A") in b.log)
    c = submit(m, "process", "C")
    time.sleep(1.2)
    assert ("cpu", "C") not in b.log and c.state == "queued"      # cpu_jobs = 1
    prev = submit(m, "process", "prev", task="maskpreview")       # the mask editor's preview doesn't wait
    wait_for(lambda: ("cpu", "prev") in b.log)
    for e in ("A:cpu1", "A:gpu", "A:cpu2", "prev:cpu1", "prev:gpu", "prev:cpu2", "C:cpu1", "C:gpu", "C:cpu2"):
        b.ev(e).set()
    wait_for(lambda: all(j.state == "completed" for j in (a, c, prev)))
    wait_for(lambda: not m.busy())          # the pool frees its place just after the job reports completed


def test_cancel_while_waiting_drops_the_request(worker):
    m, b = worker
    t = submit(m, "train", "T")
    wait_for(lambda: ("train-start", "T") in b.log)
    m.free = 0
    b.ev("P:cpu1").set()
    p = submit(m, "process", "P")
    wait_for(lambda: m.pending)
    p.cancel_event.set()
    m.gpu_release(p.id)                                           # what the backend does when the job ends
    assert not m.pending and not m.leases
    assert t.state != "failed"


def test_no_cuda_gpu_never_waits(tmp_path):
    pytest.importorskip("fastapi")
    from splatworker.config import ServerSettings
    from splatworker.server import JobManager
    from splatworker.slots import Slot

    m = JobManager(ServerSettings(data_dir=tmp_path, mdns=False), LeaseBackend(),
                   [Slot(key="vk:0000:03:00.0", vulkan=0)])        # AMD with the Vulkan trainer, no CUDA
    home = m._home_slot()
    assert home.key is None and home.cuda is None
    got = []
    from splatworker.server import Job

    job = Job(tmp_path, {"kind": "process", "datasetId": "d"}, {"jobId": "p1", "state": "running"})
    m.jobs["p1"] = job
    m.gpu_request(job, home, "features", 3000, lambda: got.append(1))
    assert got == [1] and not m.leases


def test_info_and_settings(tmp_path):
    from splatworker.cli import build_parser
    from splatworker.gui.formatting import job_view
    from splatworker.gui.paths import AppPaths
    from splatworker.gui.settings import AppSettings

    s = AppSettings(cpu_jobs=2)
    paths = AppPaths(config_dir=tmp_path / "c", data_root=tmp_path / "d", cache_dir=tmp_path / "k")
    assert build_parser().parse_args(["serve", *s.serve_args(paths)]).cpu_jobs == 2
    assert "cpu_jobs" in AppSettings(cpu_jobs=0).field_errors()
    v = job_view({"jobId": "p", "kind": "process", "state": "running", "step": 300, "totalSteps": 1000,
                  "message": "features: waiting for the GPU", "resource": "waiting-gpu", "gpu": "RTX 4080"})
    assert "waiting for the GPU" in v.detail and "RTX 4080" not in v.detail
    v = job_view({"jobId": "p", "kind": "process", "state": "running", "message": "mapping", "resource": "cpu"})
    assert "running on the CPU" in v.detail


def test_real_job_process_leases_around_gpu_stages(tmp_path, fake_splatprep):
    """A processing job through SubprocessBackend (a spawned job process, the fake splatprep's "features" then
    "mapping" stages): it asks for the GPU before features and hands it back for mapping."""
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    from test_api import AUTH, TOKEN, _images_zip, upload, wait_state

    from splatworker.backends import SubprocessBackend
    from splatworker.config import ServerSettings
    from splatworker.server import create_app
    from splatworker.slots import Slot

    settings = ServerSettings(data_dir=tmp_path / "data", token=TOKEN, mdns=False, max_upload_bytes=5 * 2**20)
    client = TestClient(create_app(settings, backend=SubprocessBackend(), devices=[Slot(key=0, cuda=0, name="GPU")]))
    with client:
        ds = upload(client, _images_zip()).json()["datasetId"]
        r = client.post("/v1/process", headers=AUTH, json={
            "datasetId": ds, "request": {"task": "sfm", "imagesDir": "images", "outputDir": "out",
                                         "features": "superpoint_lightglue", "priorPoses": "priors.json"}})
        s = wait_state(client, r.json()["jobId"], ("completed", "failed"), timeout=60)
        log = client.get(f"/v1/jobs/{s['jobId']}/log", headers=AUTH).text
        info = client.get("/v1/info", headers=AUTH).json()
    assert s["state"] == "completed", log
    assert "starting on the CPU (GPU stages on GPU 0)" in log
    assert log.index("features: asks for GPU 0") < log.index("GPU granted after") < log.index("GPU handed back")
    assert "(the GPU is free)" in log
    assert s["resource"] == "cpu"
    assert info["cpuJobs"] == {"limit": 1, "running": 0, "leases": {}, "waitingForGpu": []}
