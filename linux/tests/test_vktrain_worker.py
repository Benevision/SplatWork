"""The Vulkan trainer in the worker (splatworker/vktrain.py), without Vulkan: which trainer is chosen, the job
runner against a fake ``splatwork-vk-train`` (a Python script speaking the same JSON lines and stdin commands),
the backend's capabilities, ``/v1/info`` and the setup assistant's wording. The real program runs in
test_vktrain.py (needs a Vulkan device)."""
from __future__ import annotations

import json
import os
import queue
import stat
import sys
import threading
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from splatworker import vktrain  # noqa: E402

GPU_INFO = {"ok": True, "device": "AMD Radeon RX 6700 XT (RADV NAVI22)", "driver": "radv Mesa 25.1",
            "vendorID": 0x1002, "deviceID": 0x73DF, "deviceType": "discrete", "memoryMB": 12272,
            "interface": vktrain.INTERFACE, "subgroupSize": 64, "nativeAtomics": True}


@pytest.fixture(autouse=True)
def clean_cache():
    vktrain._info_cache.clear()
    yield
    vktrain._info_cache.clear()


def patch(monkeypatch, info=GPU_INFO, nvidia=False, gsplat=False):
    monkeypatch.setattr(vktrain, "device_info", lambda *a, **k: dict(info))
    monkeypatch.setattr(vktrain, "_nvidia_present", lambda: nvidia)
    import splatworker.gpu as gpu

    monkeypatch.setattr(gpu, "training_available", lambda: gsplat)


# ------------------------------------------------------------------ choice

def test_auto_keeps_gsplat_on_nvidia(monkeypatch):
    patch(monkeypatch, nvidia=True, gsplat=True)
    assert vktrain.select("auto")["trainer"] == "gsplat"
    patch(monkeypatch, nvidia=True, gsplat=False)            # NVIDIA without the CUDA runtime: still no Vulkan
    assert vktrain.select("auto")["trainer"] is None


def test_auto_uses_vulkan_without_nvidia(monkeypatch):
    patch(monkeypatch)
    c = vktrain.select("auto")
    assert c["trainer"] == "vulkan"
    d = vktrain.describe(c)
    assert d["kind"] == "vulkan" and d["experimental"] and d["memoryMB"] == 12272


def test_auto_falls_back_to_gsplat_and_refuses_cpu_drivers(monkeypatch):
    lavapipe = {**GPU_INFO, "device": "llvmpipe (LLVM 19)", "deviceType": "cpu"}
    patch(monkeypatch, info=lavapipe, gsplat=True)           # e.g. the experimental ROCm training runtime
    assert vktrain.select("auto")["trainer"] == "gsplat"
    patch(monkeypatch, info=lavapipe)
    c = vktrain.select("auto")
    assert c["trainer"] is None and "CPU Vulkan driver" in c["reason"]
    monkeypatch.setenv(vktrain.ALLOW_CPU_ENV, "1")
    assert vktrain.select("auto")["trainer"] == "vulkan"


def test_explicit_choices(monkeypatch):
    patch(monkeypatch, nvidia=True, gsplat=True)
    assert vktrain.select("vulkan")["trainer"] == "vulkan"   # opt-in on NVIDIA
    patch(monkeypatch, gsplat=False)
    assert vktrain.select("gsplat")["trainer"] is None
    patch(monkeypatch, info={"ok": False, "error": "no Vulkan device found"})
    c = vktrain.select("vulkan")
    assert c["trainer"] is None and "no Vulkan device" in c["reason"]
    patch(monkeypatch, info={**GPU_INFO, "interface": vktrain.INTERFACE + 1})
    assert "interface" in vktrain.select("vulkan")["reason"]


def test_gpu_entry_for_info(monkeypatch):
    patch(monkeypatch)
    import splatworker.gpu as gpu

    monkeypatch.setattr(gpu, "adapters", lambda sys_root=None: [
        {"vendor": "amd", "name": "Radeon RX 6700 XT", "memoryMB": 12288, "freeMemoryMB": 11000}])
    e = vktrain.gpu_entry()
    assert e == {"name": GPU_INFO["device"], "memoryMB": 12288, "freeMemoryMB": 11000, "vendor": "amd"}


def test_device_info_runs_the_program(tmp_path, monkeypatch):
    prog = tmp_path / "splatwork-vk-train"
    prog.write_text("#!/bin/sh\necho 'device: x' >&2\necho '" + json.dumps({k: v for k, v in GPU_INFO.items() if k != "ok"}) + "'\n")
    prog.chmod(prog.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv(vktrain.BINARY_ENV, str(prog))
    info = vktrain.device_info()
    assert info["ok"] is True and info["deviceType"] == "discrete"
    bad = tmp_path / "bad"
    bad.write_text("#!/bin/sh\necho 'no Vulkan device found' >&2\nexit 1\n")
    bad.chmod(0o755)
    assert vktrain.device_info(str(bad)) == {"ok": False, "error": "no Vulkan device found"}
    monkeypatch.setenv(vktrain.BINARY_ENV, str(tmp_path / "missing"))
    assert vktrain.binary_path() is None and not vktrain.device_info()["ok"]


# ------------------------------------------------------------- job runner

FAKE = r'''
import json, os, sys, threading, time
args = sys.argv[1:]
job = args[args.index("--job-dir") + 1]
out = args[args.index("--out") + 1]
cfg = json.load(open(args[args.index("--config") + 1]))
resumed = "--resume" in args
def emit(**m):
    print(json.dumps(m), flush=True)
cmds = []
threading.Thread(target=lambda: [cmds.append(l.strip()) for l in sys.stdin], daemon=True).start()
print("device: fake", file=sys.stderr, flush=True)
if cfg.get("fail"):
    emit(type="error", message="invalid config: strategy must be mcmc or adaptive"); sys.exit(2)
if cfg.get("crash"):
    print("Segmentation fault", file=sys.stderr, flush=True); sys.exit(139)
emit(type="status", state="loading", message="loading dataset", totalSteps=cfg["iterations"])
start = 50 if resumed else 0
emit(type="log", message="resumed" if resumed else "fresh")
for step in range(start + 1, cfg["iterations"] + 1):
    time.sleep(0.01)
    if "cancel" in cmds:
        emit(type="cancelled"); sys.exit(4)
    if "checkpoint" in cmds:
        cmds.remove("checkpoint"); open(os.path.join(job, "checkpoint.ply"), "w").write("ply"); emit(type="checkpoint", step=step)
    if "pause" in cmds:
        open(os.path.join(job, "resume.vkstate"), "w").write("state")
        emit(type="paused", step=step, resumeFile="x"); sys.exit(3)
    if step % 10 == 0:
        emit(type="status", step=step, loss=0.1, psnr=20.0, numGaussians=1000)
    if step == 20:
        emit(type="preview", step=step)
if resumed:
    os.remove(os.path.join(job, "resume.vkstate"))
open(out, "w").write("ply")
emit(type="done", numGaussians=1000, psnr=24.0, steps=cfg["iterations"] - start)
'''


@pytest.fixture
def fake(tmp_path, monkeypatch):
    prog = tmp_path / "fake-vk-train"
    prog.write_text(f"#!{sys.executable}\n{FAKE}")
    prog.chmod(0o755)
    monkeypatch.setenv(vktrain.BINARY_ENV, str(prog))
    monkeypatch.setattr(vktrain, "fill_depth_priors", lambda *a, **k: None)
    return prog


def run_job(job: Path, config: dict, act=None, timeout=30.0):
    """subprocess_entry in a thread; ``act(kind, payload, events)`` reacts to messages. Returns (msgs, outcome)."""
    q: "queue.Queue" = queue.Queue()
    ev = {k: threading.Event() for k in ("cancel", "pause", "ckpt")}
    t = threading.Thread(target=vktrain.subprocess_entry,
                         args=("/nonexistent", config, str(job), None, q, ev["cancel"], ev["pause"], ev["ckpt"]))
    t.start()
    msgs = []
    while True:
        kind, payload = q.get(timeout=timeout)
        msgs.append((kind, payload))
        if act:
            act(kind, payload, ev)
        if kind in ("done", "paused", "cancelled", "error"):
            t.join(10)
            return msgs, (kind, payload)


def test_runner_trains_and_forwards_messages(fake, tmp_path):
    job = tmp_path / "job"
    job.mkdir()
    msgs, out = run_job(job, {"iterations": 40})
    assert out[0] == "done" and out[1]["trainer"] == "vulkan" and out[1]["psnr"] == 24.0
    kinds = [k for k, _ in msgs]
    assert ("preview", 20) in msgs and kinds.count("status") >= 4
    assert ("log", "device: fake") in msgs and ("log", "fresh") in msgs
    assert json.loads((job / "vk-config.json").read_text()) == {"iterations": 40}
    assert (job / "result.ply").is_file()


def test_runner_checkpoint_pause_resume(fake, tmp_path):
    job = tmp_path / "job"
    job.mkdir()

    def act(kind, payload, ev):
        if kind == "status" and payload.get("step") == 20:
            ev["ckpt"].set()
        if kind == "checkpoint":
            ev["pause"].set()
    msgs, out = run_job(job, {"iterations": 400}, act)
    assert out[0] == "paused" and out[1] > 20
    assert any(k == "checkpoint" for k, _ in msgs) and (job / "checkpoint.ply").is_file()
    assert (job / "resume.vkstate").is_file()
    assert "--resume" in vktrain.command(str(fake), "/d", job, job / "c.json")
    msgs, out = run_job(job, {"iterations": 60})
    assert out[0] == "done" and ("log", "resumed") in msgs
    assert not (job / "resume.vkstate").exists()


def test_runner_cancel_error_and_crash(fake, tmp_path):
    for name, cfg, act, expect in [
        ("c", {"iterations": 5000}, lambda k, p, ev: ev["cancel"].set() if k == "status" else None, "cancelled"),
        ("e", {"iterations": 10, "fail": True}, None, "error"),
        ("x", {"iterations": 10, "crash": True}, None, "error"),
    ]:
        job = tmp_path / name
        job.mkdir()
        msgs, out = run_job(job, cfg, act)
        assert out[0] == expect, (name, msgs)
        if name == "e":
            assert "invalid config" in out[1]
        if name == "x":
            assert "exit code 139" in out[1] and "Segmentation fault" in out[1]


def test_runner_without_program(tmp_path, monkeypatch):
    monkeypatch.setenv(vktrain.BINARY_ENV, str(tmp_path / "missing"))
    _, out = run_job(tmp_path, {"iterations": 10})
    assert out[0] == "error" and "not installed" in out[1]


# ---------------------------------------------------------- backend, info

def test_backend_routes_train_and_census(monkeypatch):
    from splatworker.backends import SubprocessBackend

    patch(monkeypatch)
    import splatworker.processing as processing

    monkeypatch.setattr(processing, "splatprep_available", lambda: False)
    monkeypatch.setattr(processing, "frames_available", lambda: False)
    b = SubprocessBackend(trainer="auto")
    assert b.capabilities() == ["train"] and b.trainer_choice()["trainer"] == "vulkan"
    assert not b.census_available()                   # Auto clean renders with gsplat
    patch(monkeypatch, nvidia=True, gsplat=False)
    assert SubprocessBackend(trainer="auto").capabilities() == []


def test_info_reports_the_trainer(monkeypatch, tmp_path):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from splatworker.backends import SubprocessBackend
    from splatworker.config import ServerSettings
    from splatworker.server import create_app
    import splatworker.processing as processing
    import splatworker.server as server

    patch(monkeypatch)
    monkeypatch.setattr(processing, "splatprep_available", lambda: False)
    monkeypatch.setattr(processing, "frames_available", lambda: False)
    monkeypatch.setattr(server, "list_gpus", lambda *a, **k: [], raising=False)
    import splatworker.gpu as gpu

    monkeypatch.setattr(gpu, "list_gpus", lambda *a, **k: [])
    settings = ServerSettings(data_dir=tmp_path, mdns=False, trainer="auto")
    app = create_app(settings, backend=SubprocessBackend(trainer="auto"), devices=[None])
    with TestClient(app) as c:
        info = c.get("/v1/info").json()
    assert info["trainer"]["kind"] == "vulkan" and info["trainer"]["experimental"]
    assert "train" in info["capabilities"] and "pause" in info["capabilities"]
    assert "census" not in info["capabilities"]
    assert info["gpus"] and info["gpus"][0]["vendor"] == "amd"
    assert info["backend"].startswith("SplatWork Vulkan trainer (experimental)")


def test_server_resume_files():
    from splatworker.server import _drop_resume, _has_resume

    import tempfile

    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        assert not _has_resume(d)
        (d / "resume.vkstate").write_text("x")
        assert _has_resume(d)
        _drop_resume(d)
        assert not _has_resume(d)


def test_app_setting_passes_the_trainer():
    from splatworker.gui.paths import AppPaths
    from splatworker.gui.settings import AppSettings

    paths = AppPaths.from_env()
    assert "--trainer" not in AppSettings().serve_args(paths)
    assert AppSettings(trainer="vulkan").serve_args(paths)[-2:] == ["--trainer", "vulkan"]
    assert AppSettings.from_dict({"trainer": "bogus"}).trainer == "auto"


# ------------------------------------------------------------- setup wording

def test_setup_offers_vulkan_training_on_amd(tmp_path):
    from splatworker.gpu_vendors import Adapter
    from splatworker.gui.nvidia import DriverInfo
    from splatworker.gui.runtime_plan import CPU_PROFILE_ID, choose_profile

    amd = Adapter(vendor="amd", vendor_id=0x1002, device_id=0x73DF, pci="0000:03:00.0", driver="amdgpu",
                  name="AMD Radeon RX 6700 XT", memory_mb=12288, gfx="gfx1031")
    without = DriverInfo(adapters=[amd], kfd=True)
    c = choose_profile(without)
    assert not c.can_train and c.blocking
    with_vk = DriverInfo(adapters=[amd], kfd=True, vulkan=dict(GPU_INFO))
    assert with_vk.vulkan_trains
    c = choose_profile(with_vk)
    assert c.can_train and c.blocking is None and c.profile_id != CPU_PROFILE_ID
    assert "Vulkan trainer" in c.summary and any("hasn't been compared" in w for w in c.warnings)
    no_kfd = DriverInfo(adapters=[amd], kfd=False, vulkan=dict(GPU_INFO))
    c = choose_profile(no_kfd)
    assert c.can_train and c.profile_id == CPU_PROFILE_ID
    title, text = with_vk.problem()
    assert "Vulkan trainer" in text and "experimental" in text
