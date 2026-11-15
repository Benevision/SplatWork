"""Job slots per GPU (splatworker/slots.py): CUDA GPUs, GPUs only the Vulkan trainer can use, and slow integrated
GPUs that must not hold up a run — with fake GPU lists (BensPC: an RTX 4080 and a Radeon 610M; an AMD-only PC; a
laptop with only an integrated GPU)."""
from __future__ import annotations

import json
import stat
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from splatworker import slots as S  # noqa: E402
from splatworker import vktrain  # noqa: E402

RTX = {"index": 0, "name": "NVIDIA GeForce RTX 4080", "vendor": "nvidia", "memoryMB": 16376, "freeMemoryMB": 15000,
       "pci": "0000:01:00.0"}
VK_RTX = {"index": 0, "name": "NVIDIA GeForce RTX 4080", "deviceType": "discrete", "vendorID": 0x10DE,
          "memoryMB": 16376, "pci": "0000:01:00.0"}
VK_610M = {"index": 1, "name": "AMD Radeon 610M (RADV GFX1036)", "deviceType": "integrated", "vendorID": 0x1002,
           "memoryMB": 2048, "pci": "0000:c5:00.0"}
VK_LAVAPIPE = {"index": 2, "name": "llvmpipe (LLVM 19.1.7, 256 bits)", "deviceType": "cpu", "vendorID": 0x10005,
               "memoryMB": 15701, "pci": None}
VK_6700 = {"index": 0, "name": "AMD Radeon RX 6700 XT (RADV NAVI22)", "deviceType": "discrete", "vendorID": 0x1002,
           "memoryMB": 12272, "pci": "0000:03:00.0"}
BENSPC = [VK_RTX, VK_610M, VK_LAVAPIPE]


@pytest.fixture(autouse=True)
def no_measurements(tmp_path, monkeypatch):
    monkeypatch.setenv("SPLATWORK_GPU_SPEEDS", str(tmp_path / "gpu-speeds.json"))


def test_benspc_default_keeps_the_610m_out():
    slots = S.plan(None, [RTX], BENSPC, "vulkan")
    assert [(s.key, s.cuda, s.vulkan, s.slow) for s in slots] == [(0, 0, 0, False)]


def test_benspc_with_the_610m_ticked():
    slots = S.plan([0, "vk:0000:c5:00.0"], [RTX], BENSPC, "vulkan")
    assert [(s.key, s.cuda, s.vulkan, s.integrated, s.slow) for s in slots] == [
        (0, 0, 0, False, False), ("vk:0000:c5:00.0", None, 1, True, True)]
    assert slots[1].id == "vk:0000:c5:00.0" and slots[1].vendor == "amd"
    assert slots[1].takes_kind("train") and not slots[1].takes_kind("process") and not slots[1].takes_kind("census")
    assert slots[0].info()["kinds"] == ["train", "process", "census"] and slots[1].info()["slow"]


def test_gsplat_trainer_has_cuda_slots_only():
    slots = S.plan([0, "vk:0000:c5:00.0"], [RTX], BENSPC, "gsplat")
    assert [(s.key, s.vulkan) for s in slots] == [(0, None)]


def test_amd_only_pc_and_integrated_only_laptop():
    slots = S.plan(None, [], [VK_6700, {**VK_LAVAPIPE, "index": 1}], "vulkan")
    assert [(s.key, s.vulkan, s.slow) for s in slots] == [("vk:0000:03:00.0", 0, False)]
    laptop = S.plan(None, [], [{**VK_610M, "index": 0}], "vulkan")      # the only GPU: on, and not "slow"
    assert [(s.vulkan, s.integrated, s.slow) for s in laptop] == [(0, True, False)]
    assert S.plan(None, [], [], "vulkan")[0].key is None                 # nothing found: the program's own pick
    assert S.plan(None, [], [VK_LAVAPIPE], "vulkan")[0].key is None      # CPU drivers aren't slots


def test_cpu_driver_counts_when_allowed(monkeypatch):
    monkeypatch.setenv("SPLATWORK_VK_ALLOW_CPU", "1")
    slots = S.plan(None, [], [{**VK_LAVAPIPE, "index": 0}], "vulkan")
    assert [(s.vulkan, s.slow) for s in slots] == [(0, False)]


def test_matching_by_name_without_pci_and_unlisted_indices_kept():
    rtx = {k: v for k, v in RTX.items() if k != "pci"}
    vk = [{**VK_610M, "index": 0}, {**VK_RTX, "index": 1, "pci": None}]
    slots = S.plan(None, [rtx], vk, "vulkan")
    assert [(s.key, s.vulkan) for s in slots] == [(0, 1)]
    assert [s.key for s in S.plan([1], [], [], "gsplat")] == [1]         # as before: chosen but not listed


def test_keys_and_choices():
    assert S.parse_key("0") == 0 and S.parse_key(" vk:0000:c5:00.0") == "vk:0000:c5:00.0"
    with pytest.raises(ValueError):
        S.parse_key("gpu1")
    assert S.is_slot_key(0) and S.is_slot_key("vk:x") and not S.is_slot_key(True) and not S.is_slot_key(-1)
    rows = S.gpu_choices([RTX], BENSPC, None)
    assert [(r["key"], r["active"], r["integrated"]) for r in rows] == [(0, True, False),
                                                                        ("vk:0000:c5:00.0", False, True)]
    assert "not measured yet" in rows[1]["subtitle"] and rows[0]["title"] == "GPU 0: NVIDIA GeForce RTX 4080"
    rows = S.gpu_choices([RTX], BENSPC, [0, "vk:0000:c5:00.0"])
    assert [r["active"] for r in rows] == [True, True]


def test_cli_and_app_settings_accept_vulkan_keys(tmp_path):
    from splatworker.cli import _parse_gpus, build_parser
    from splatworker.gui.paths import AppPaths
    from splatworker.gui.settings import AppSettings

    assert _parse_gpus("0,vk:0000:c5:00.0") == [0, "vk:0000:c5:00.0"]
    assert _parse_gpus("all") is None
    s = AppSettings(gpus=[0, "vk:0000:c5:00.0"])
    assert "gpus" not in s.field_errors()
    paths = AppPaths(config_dir=tmp_path / "c", data_root=tmp_path / "d", cache_dir=tmp_path / "k")
    args = build_parser().parse_args(["serve", *s.serve_args(paths)])
    assert args.gpus == [0, "vk:0000:c5:00.0"]
    assert "gpus" in AppSettings(gpus=["gpu1"]).field_errors()


def test_command_passes_the_slot_device(tmp_path, monkeypatch):
    monkeypatch.delenv(vktrain.DEVICE_ENV, raising=False)
    cmd = vktrain.command("vk", "ds", tmp_path, tmp_path / "c.json", 1)
    assert cmd[cmd.index("--device") + 1] == "1"
    assert "--device" not in vktrain.command("vk", "ds", tmp_path, tmp_path / "c.json")
    monkeypatch.setenv(vktrain.DEVICE_ENV, "2")
    cmd = vktrain.command("vk", "ds", tmp_path, tmp_path / "c.json")
    assert cmd[cmd.index("--device") + 1] == "2"


@pytest.mark.skipif(sys.platform == "win32", reason="fake programs are POSIX scripts")
def test_list_devices_runs_the_program(tmp_path, monkeypatch):
    prog = tmp_path / "splatwork-vk-train"
    prog.write_text("#!/bin/sh\necho '" + json.dumps({"devices": BENSPC, "interface": 1}) + "'\n")
    prog.chmod(prog.stat().st_mode | stat.S_IEXEC)
    vktrain._list_cache.clear()
    out = vktrain.list_devices(str(prog))
    assert out["ok"] and [d["index"] for d in out["devices"]] == [0, 1, 2]
    bad = tmp_path / "bad"
    bad.write_text("#!/bin/sh\necho 'unknown argument --list-devices' >&2\nexit 2\n")
    bad.chmod(0o755)
    assert vktrain.list_devices(str(bad)) == {"ok": False, "devices": [], "error": "unknown argument --list-devices"}
    vktrain._list_cache.clear()


# ----------------------------------------------------------- scheduling

def _manager(tmp_path, slot_list):
    pytest.importorskip("fastapi")
    from splatworker.config import ServerSettings
    from splatworker.server import Job, JobManager

    settings = ServerSettings(data_dir=tmp_path, mdns=False)
    m = JobManager(settings, backend=None, devices=slot_list)

    def job(kind="train", **spec):
        return Job(tmp_path, {"kind": kind, "datasetId": "d", **spec}, {"state": "queued"})
    return m, job


def test_slow_slot_takes_only_standalone_train_jobs_while_fast_ones_are_busy(tmp_path):
    fast, slow = S.plan([0, "vk:0000:c5:00.0"], [RTX], BENSPC, "vulkan")
    m, job = _manager(tmp_path, [fast, slow])
    train, block = job(config={}), job(config={"region": {"min": [0, 0, 0], "max": [1, 1, 1]}})
    step, proc = job(config={}, parentJobId="w1"), job("process", request={})
    assert all(m.slot_takes(fast, j) for j in (train, block, step, proc))
    assert not any(m.slot_takes(slow, j) for j in (train, block, step, proc))   # the 4080 is free
    m.running[fast.key] = "j0"
    assert m.slot_takes(slow, train)
    assert not m.slot_takes(slow, block) and not m.slot_takes(slow, step) and not m.slot_takes(slow, proc)
    assert not m.slot_takes(slow, job(config={"cameraIds": [1, 2]}))


def test_named_slot_and_old_style_devices(tmp_path):
    fast, slow = S.plan([0, "vk:0000:c5:00.0"], [RTX], BENSPC, "vulkan")
    m, job = _manager(tmp_path, [fast, slow])
    named = job(config={}, slot="vk:0000:c5:00.0")
    assert m.slot_takes(slow, named) and not m.slot_takes(fast, named)
    m2, _ = _manager(tmp_path, [0, 1])                     # plain CUDA indices still work
    assert [s.cuda for s in m2.slots] == [0, 1] and m2.devices == [0, 1]
    assert [s.key for s in _manager(tmp_path, None)[0].slots] == [None]


def test_info_lists_slots_and_train_jobs_may_name_one(tmp_path, monkeypatch):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    import splatworker.gpu as gpu
    import splatworker.processing as processing
    from splatworker.backends import SubprocessBackend
    from splatworker.config import ServerSettings
    from splatworker.server import create_app

    monkeypatch.setattr(vktrain, "device_info", lambda *a, **k: {"ok": True, "device": "NVIDIA GeForce RTX 4080",
                                                                  "deviceType": "discrete", "vendorID": 0x10DE,
                                                                  "memoryMB": 16376, "interface": vktrain.INTERFACE})
    monkeypatch.setattr(vktrain, "list_devices", lambda *a, **k: {"ok": True, "devices": BENSPC})
    monkeypatch.setattr(vktrain, "_nvidia_present", lambda: True)
    monkeypatch.setattr(gpu, "training_available", lambda: True)
    monkeypatch.setattr(gpu, "list_gpus", lambda *a, **k: [RTX])
    monkeypatch.setattr(processing, "splatprep_available", lambda: False)
    monkeypatch.setattr(processing, "frames_available", lambda: False)
    settings = ServerSettings(data_dir=tmp_path, mdns=False, trainer="vulkan", gpus=[0, "vk:0000:c5:00.0"])
    app = create_app(settings, backend=SubprocessBackend(trainer="vulkan"))
    with TestClient(app) as c:
        info = c.get("/v1/info").json()
        r = c.post("/v1/jobs", json={"datasetId": "nope", "config": {}, "slot": "vk:other"})
    assert [g["name"] for g in info["gpus"]] == ["NVIDIA GeForce RTX 4080"]      # the 610M isn't a device
    assert [(s["id"], s["vulkan"], s["slow"], s["kinds"]) for s in info["slots"]] == [
        ("cuda:0", 0, False, ["train", "process", "census"]), ("vk:0000:c5:00.0", 1, True, ["train"])]
    assert r.status_code in (400, 404)


# ---------------------------------------------------------- measured speeds

def _speeds(**sps):
    from splatworker import gpuspeed

    devs = {"rtx": VK_RTX, "m610": VK_610M, "rx": VK_6700}
    return {gpuspeed.device_key(devs[k]): {"name": devs[k]["name"], "stepsPerSecond": v} for k, v in sps.items()}


def test_slow_comes_from_measured_speeds():
    sel = [0, "vk:0000:c5:00.0"]
    # not measured: the integrated GPU counts as slow
    assert [s.slow for s in S.plan(sel, [RTX], BENSPC, "vulkan")] == [False, True]
    # only one measured: still the conservative guess
    assert [s.slow for s in S.plan(sel, [RTX], BENSPC, "vulkan", _speeds(m610=3.0))] == [False, True]
    fast, slow = S.plan(sel, [RTX], BENSPC, "vulkan", _speeds(rtx=60.0, m610=1.2))
    assert slow.slow and slow.speed == 1.2 and slow.relative == 0.02 and fast.relative == 1.0
    assert slow.info()["stepsPerSecond"] == 1.2 and slow.info()["relativeSpeed"] == 0.02
    # measured fast enough (≥ 25 % of the fastest): an ordinary slot, even integrated
    assert [s.slow for s in S.plan(sel, [RTX], BENSPC, "vulkan", _speeds(rtx=40.0, m610=12.0))] == [False, False]
    # a slow discrete card beside a fast one is slow too, once measured
    rx = {**VK_6700, "index": 1}
    two = S.plan(None, [RTX], [VK_RTX, rx], "vulkan", _speeds(rtx=100.0, rx=10.0))
    assert [(s.key, s.slow) for s in two] == [(0, False), ("vk:0000:03:00.0", True)]


def test_measured_speeds_in_the_settings_rows():
    rows = S.gpu_choices([RTX], BENSPC, None, _speeds(rtx=60.0, m610=1.2))
    assert rows[0]["subtitle"].startswith("16 GB · Measured 60.0 training steps/s (100% of the fastest)")
    assert "Measured 1.2 training steps/s (2% of the fastest). Much slower" in rows[1]["subtitle"]
    rows = S.gpu_choices([RTX], BENSPC, None, _speeds(rtx=40.0, m610=12.0))
    assert "Fast enough" in rows[1]["subtitle"]


def test_speed_file_round_trip_and_server_refresh(tmp_path, monkeypatch):
    from splatworker import gpuspeed

    path = tmp_path / "gpu-speeds.json"
    assert gpuspeed.load() == {}
    gpuspeed.save(_speeds(rtx=60.0))
    gpuspeed.save(_speeds(m610=1.2))                     # merged, not replaced
    assert set(gpuspeed.load()) == set(_speeds(rtx=1, m610=1))
    other_driver = {**VK_610M, "driver": "radv Mesa 26.0"}
    assert gpuspeed.speed_of(other_driver, gpuspeed.load()) is None    # a new driver: measure again
    path.write_text("not json")
    assert gpuspeed.load() == {}
    gpuspeed.save(_speeds(rtx=60.0, m610=1.2))
    fast, slow = S.plan([0, "vk:0000:c5:00.0"], [RTX], BENSPC, "vulkan")
    m, job = _manager(tmp_path, [fast, slow])
    m.refresh_speeds()
    assert slow.slow and slow.relative == 0.02
    gpuspeed.save(_speeds(m610=30.0))                    # re-measured: picked up without a restart
    import os
    os.utime(path, ns=(1, 2_000_000_000_000_000_000))
    m.refresh_speeds()
    assert not slow.slow


def test_config_holds_the_splat_count():
    from splatworker import gpuspeed

    cfg = gpuspeed.config()
    assert cfg["iterations"] == 300 and cfg["densifyStart"] > 300 and cfg["maxGaussians"] == 200_000
