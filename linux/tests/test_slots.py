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
    # the CUDA slot still knows its Vulkan device (a job may ask for the Vulkan trainer, 1.7.9)
    assert [(s.key, s.vulkan, s.integrated) for s in slots] == [(0, 0, False)]


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


def test_slower_slot_starts_waiting_train_jobs_while_fast_ones_are_busy(tmp_path):
    fast, slow = S.plan([0, "vk:0000:c5:00.0"], [RTX], BENSPC, "vulkan")
    m, job = _manager(tmp_path, [fast, slow])
    small = {"maxGaussians": 200_000}
    train, block = job(config=small), job(config={**small, "region": {"min": [0, 0, 0], "max": [1, 1, 1]}})
    step, proc = job(config=small, parentJobId="w1"), job("process", request={})
    assert all(m.slot_takes(fast, j) for j in (train, block, step, proc))
    assert not any(m.slot_takes(slow, j) for j in (train, block, step, proc))   # the 4080 is free
    m.running[fast.key] = "j0"
    # 1.7.8: block jobs and pipeline steps too (they move to the 4080 when it frees up); never processing
    assert m.slot_takes(slow, train) and m.slot_takes(slow, block) and m.slot_takes(slow, step)
    assert not m.slot_takes(slow, proc)
    assert not m.slot_takes(slow, job(config={"maxGaussians": 50_000_000}))     # doesn't fit its memory


def test_always_use_makes_a_slower_gpu_an_ordinary_slot(tmp_path):
    sel = [0, "vk:0000:c5:00.0"]
    fast, slow = S.plan(sel, [RTX], BENSPC, "vulkan", modes={"vk:0000:c5:00.0": "always"})
    assert slow.slower and not slow.slow and slow.user_choice == "always"
    assert slow.info()["slow"] is False and slow.info()["userChoice"] == "always" and slow.info()["slower"]
    m, job = _manager(tmp_path, [fast, slow])
    assert m.slot_takes(slow, job(config={"maxGaussians": 200_000, "region": {}}))   # even with the 4080 free


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
    vk = {d["index"]: d for d in info["vulkanDevices"]}
    assert set(vk) == {0, 1} and vk[0]["slot"] == "cuda:0" and vk[1]["slot"] == "vk:0000:c5:00.0"
    from splatworker import gpuspeed

    gpuspeed.save(_speeds(rtx=251.5, m610=4.0))
    unticked = ServerSettings(data_dir=tmp_path / "b", mdns=False, trainer="vulkan", gpus=[0])
    with TestClient(create_app(unticked, backend=SubprocessBackend(trainer="vulkan"))) as c:
        info = c.get("/v1/info").json()
    vk = {d["index"]: d for d in info["vulkanDevices"]}
    assert vk[1]["slot"] is None and vk[1]["stepsPerSecond"] == 4.0 and vk[1]["size"] == "standard"
    assert vk[0]["stepsPerSecond"] == 251.5 and [s["id"] for s in info["slots"]] == ["cuda:0"]


# ---------------------------------------------------------- measured speeds

def _speeds(size="standard", **sps):
    from splatworker import gpuspeed

    devs = {"rtx": VK_RTX, "m610": VK_610M, "rx": VK_6700}
    return {gpuspeed.device_key(devs[k]): {"name": devs[k]["name"], "stepsPerSecond": v, "recipe": gpuspeed.RECIPE,
                                           "size": size, "peakMemoryMB": 3072}
            for k, v in sps.items()}


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
    assert rows[0]["subtitle"].startswith("16 GB · Measured 60.0 training steps/s on the standard scene (100% of "
                                          "the fastest). Used 3.0 GB.")
    assert ("Measured 1.2 training steps/s on the standard scene (2% of the NVIDIA GeForce RTX 4080). Used 3.0 GB. "
            "Slower GPU") in rows[1]["subtitle"]
    assert rows[1]["slower"] and rows[1]["share"] == "about 2% of the NVIDIA GeForce RTX 4080"
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


def test_config_is_the_benchmark_scene_held_fixed():
    from splatworker import gpuspeed

    cfg = gpuspeed.config()
    assert cfg["maxGaussians"] == 1_000_000 and cfg["maxImageSize"] == 1600
    assert cfg["densifyStart"] >= cfg["iterations"] >= 1_000_000        # no densification; cancelled when timed
    assert gpuspeed.config("medium")["maxGaussians"] == 200_000


def test_old_recipe_and_different_sizes(tmp_path):
    from splatworker import gpuspeed

    old = {k: {**v, "recipe": 1} for k, v in _speeds(rtx=251.5, m610=40.0).items()}
    gpuspeed.save(old)
    assert gpuspeed.load() == {}                                      # 1.7.3 / 1.7.4's 300-step runs: unmeasured
    sel = [0, "vk:0000:c5:00.0"]
    mixed = {**_speeds(rtx=60.0), **_speeds("medium", m610=30.0)}     # the iGPU only fit the medium scene
    fast, slow = S.plan(sel, [RTX], BENSPC, "vulkan", mixed)
    assert slow.slow and slow.relative is None and slow.size == "medium" and fast.relative == 1.0
    rows = S.gpu_choices([RTX], BENSPC, None, mixed)
    assert "on the medium scene. Used 3.0 GB. Slower GPU" in rows[1]["subtitle"] and rows[1]["slower"]
    assert not gpuspeed.comparable(list(mixed.values())) and gpuspeed.comparable(list(_speeds(rtx=1, m610=1).values()))


FAKE_TRAINER = r"""
import json, sys, threading, time
args = sys.argv[1:]
if "benchmark-scene" in args[args.index("--dataset") + 1]:
    print(json.dumps({"type": "error", "message": "out of device memory"}), flush=True)
    sys.exit(1)
stop = threading.Event()
threading.Thread(target=lambda: (sys.stdin.readline(), stop.set()), daemon=True).start()
print(json.dumps({"type": "status", "message": "loading images 1/16"}), flush=True)
step = 0
while not stop.is_set():
    time.sleep(0.02)
    step += 1
    print(json.dumps({"type": "status", "step": step}), flush=True)
print(json.dumps({"type": "cancelled"}), flush=True)
sys.exit(4)
"""


@pytest.mark.skipif(sys.platform == "win32", reason="fake programs are POSIX scripts")
def test_timed_window_and_fallback_to_a_smaller_scene(tmp_path, monkeypatch):
    from splatworker import gpuspeed
    import splatworker.studio.benchmark as bench

    prog = tmp_path / "splatwork-vk-train"
    prog.write_text(f"#!{sys.executable}\n" + FAKE_TRAINER)
    prog.chmod(0o755)
    monkeypatch.setenv(vktrain.BINARY_ENV, str(prog))
    monkeypatch.setattr(gpuspeed, "WARMUP", 0.3)
    monkeypatch.setattr(gpuspeed, "WINDOW", 0.8)
    monkeypatch.setattr(bench, "build_scene", lambda root, spec, *a, **k: Path(root))
    said = []
    out = gpuspeed.measure_all([VK_610M], tmp_path / "cache", progress=said.append)
    rec = out[gpuspeed.device_key(VK_610M)]
    assert rec["size"] == "medium" and rec["splats"] == 200_000 and rec["recipe"] == gpuspeed.RECIPE
    assert rec["failedSizes"] == ["standard: out of device memory"]
    assert 10 < rec["stepsPerSecond"] < 60 and rec["windowSteps"] >= 3 and rec["windowSeconds"] >= 0.8
    assert any("Measuring AMD Radeon 610M (RADV GFX1036) (medium scene): measuring" in m for m in said)
    assert gpuspeed.load()[gpuspeed.device_key(VK_610M)]["size"] == "medium"


# ------------------------------------------- BensPC as 1.7.3 really saw it

# NVML's PCI address has an 8-digit domain; the Flatpak's Vulkan loader lists every GPU twice (two driver copies);
# RADV names the Ryzen's integrated Radeon 610M after the CPU.
NVML_RTX = {"index": 0, "name": "NVIDIA GeForce RTX 4080", "vendor": "nvidia", "memoryMB": 16376,
            "freeMemoryMB": 15000, "pci": "00000000:01:00.0"}
RADV_NAME = "AMD Ryzen 9 9950X3D 16-Core Processor (RADV RAPHAEL_MENDOCINO)"
REAL_VK = [
    {"index": 0, "name": "NVIDIA GeForce RTX 4080", "deviceType": "discrete", "vendorID": 0x10DE, "memoryMB": 16376,
     "pci": "0000:01:00.0", "driver": "NVIDIA 580.95.05", "apiVersion": 4210991, "driverVersion": 2436022336},
    {"index": 1, "name": RADV_NAME, "deviceType": "integrated", "vendorID": 0x1002, "memoryMB": 2048,
     "pci": "0000:11:00.0", "driver": "radv Mesa 25.1.3", "apiVersion": 4210991, "driverVersion": 104861699},
    {"index": 2, "name": "NVIDIA GeForce RTX 4080", "deviceType": "discrete", "vendorID": 0x10DE, "memoryMB": 16376,
     "pci": "0000:01:00.0", "driver": "NVIDIA 580.95.05", "apiVersion": 4210991, "driverVersion": 2436022336},
    {"index": 3, "name": RADV_NAME, "deviceType": "integrated", "vendorID": 0x1002, "memoryMB": 2048,
     "pci": "0000:11:00.0", "driver": "radv Mesa 25.1.3", "apiVersion": 4210991, "driverVersion": 104861699},
    {"index": 4, "name": "llvmpipe (LLVM 19.1.7, 256 bits)", "deviceType": "cpu", "vendorID": 0x10005,
     "memoryMB": 15701, "pci": None, "driver": "llvmpipe Mesa 25.1.3"},
]
REAL_ADAPTERS = [{"vendor": "nvidia", "name": "NVIDIA GeForce RTX 4080", "pci": "0000:01:00.0", "integrated": False},
                 {"vendor": "amd", "name": "AMD Radeon 610M", "pci": "0000:11:00.0", "integrated": True}]


def test_duplicate_vulkan_entries_are_one_gpu():
    devs = vktrain.tidy_devices(REAL_VK, REAL_ADAPTERS)
    assert [(d["index"], d.get("duplicates")) for d in devs] == [(0, [2]), (1, [3]), (4, None)]
    assert devs[1]["integrated"] and devs[1]["label"] == "AMD Radeon 610M (integrated)"
    assert devs[0]["label"] == "NVIDIA GeForce RTX 4080" and not devs[0]["integrated"]
    newer = [dict(REAL_VK[1]), {**REAL_VK[3], "driver": "radv Mesa 25.2.0", "driverVersion": 104865792}]
    assert [d["index"] for d in vktrain.tidy_devices(newer)] == [3]          # the newer driver copy is kept
    assert vktrain.tidy_devices(vktrain.tidy_devices(REAL_VK, REAL_ADAPTERS)) == devs   # idempotent


def test_benspc_settings_rows_as_reported():
    devs = vktrain.tidy_devices(REAL_VK, REAL_ADAPTERS)
    rows = S.gpu_choices([NVML_RTX], devs, None)                      # the server's NVML list (8-digit domain)
    assert [(r["key"], r["title"], r["active"]) for r in rows] == [
        (0, "GPU 0: NVIDIA GeForce RTX 4080", True), ("vk:0000:11:00.0", "AMD Radeon 610M (integrated)", False)]
    gui_cuda = [{"index": 0, "name": "NVIDIA GeForce RTX 4080", "memoryMB": 16376}]   # the app's nvidia-smi list
    assert [r["key"] for r in S.gpu_choices(gui_cuda, devs, None)] == [0, "vk:0000:11:00.0"]
    assert [r["key"] for r in S.gpu_choices(gui_cuda, REAL_VK, None)] == [0, "vk:0000:11:00.0"]  # raw list too


def test_benspc_slots_as_reported():
    devs = vktrain.tidy_devices(REAL_VK, REAL_ADAPTERS)
    slots = S.plan(None, [NVML_RTX], devs, "vulkan")
    assert [(s.key, s.vulkan) for s in slots] == [(0, 0)]
    slots = S.plan([0, "vk:0000:11:00.0"], [NVML_RTX], devs, "vulkan")
    assert [(s.key, s.vulkan, s.slow, s.name) for s in slots] == [
        (0, 0, False, "NVIDIA GeForce RTX 4080"), ("vk:0000:11:00.0", 1, True, "AMD Radeon 610M (integrated)")]


def test_integrated_from_the_pci_scan_and_matching_by_memory():
    odd = [{**REAL_VK[1], "deviceType": "other"}]                    # a driver that doesn't say "integrated"
    devs = vktrain.tidy_devices(odd, REAL_ADAPTERS)
    assert devs[0]["integrated"]
    assert [r["active"] for r in S.gpu_choices([NVML_RTX], devs, None)] == [True, False]
    renamed = [{**REAL_VK[0], "name": "NVIDIA GeForce RTX 4080 (driver name)", "pci": None, "uuid": "ab"}]
    assert S.match_vulkan([{"index": 0, "name": "NVIDIA GeForce RTX 4080", "memoryMB": 16376}], renamed)[0]["index"] == 0
