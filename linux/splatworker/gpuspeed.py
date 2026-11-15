"""Measured training speed of each Vulkan GPU, for the job slots (slots.py).

Instead of guessing from spec sheets, each device trains the same small synthetic scene (the standard benchmark's
scene, studio/benchmark.py, at 800×600, 16 photos, 200 k splats held fixed) for 300 steps through
``splatwork-vk-train --device N``; the program's own ``stepsPerSecond`` (training loop only) is the speed. Run on
demand — *Settings → GPUs → Measure GPU speed*, or ``splatworker measure-gpus`` — while the GPUs are idle, and cached
per device and driver in ``gpu-speeds.json`` in the app's config dir (a driver update means measuring again).
The server reads the file whenever it changes: no restart needed.
"""
from __future__ import annotations

import json
import os
import subprocess
import tempfile
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

FILE = "gpu-speeds.json"
PATH_ENV = "SPLATWORK_GPU_SPEEDS"     # tests / a custom location
SCENE_DIR = "gpu-speed-scene-v1"
STEPS = 300


def cache_path() -> Path:
    env = os.environ.get(PATH_ENV)
    if env:
        return Path(env)
    from .gui.paths import AppPaths

    return AppPaths.from_env().config_dir / FILE


def device_key(dev: Dict[str, Any]) -> str:
    """Which measurement belongs to a device: its PCI address (else name) and driver."""
    return f"{dev.get('pci') or dev.get('name')}|{dev.get('driver') or ''}"


_lock = threading.Lock()
_cache: Dict[str, Any] = {"path": None, "mtime": None, "data": {}}


def load(path: Optional[Path] = None) -> Dict[str, Dict[str, Any]]:
    """{device_key: record}; {} when nothing was measured. Re-read only when the file changed."""
    p = Path(path) if path else cache_path()
    try:
        st = p.stat()
        mtime = (st.st_mtime_ns, st.st_size)     # size too: coarse mtimes on some file systems
    except OSError:
        return {}
    with _lock:
        if _cache["path"] == str(p) and _cache["mtime"] == mtime:
            return _cache["data"]
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            data = {k: v for k, v in data.items() if isinstance(v, dict) and v.get("stepsPerSecond")}
        except (OSError, ValueError, AttributeError):
            data = {}
        _cache.update(path=str(p), mtime=mtime, data=data)
        return data


def save(records: Dict[str, Dict[str, Any]], path: Optional[Path] = None) -> None:
    p = Path(path) if path else cache_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    data = {}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        pass
    data.update(records)
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=1), encoding="utf-8")
    tmp.replace(p)


def speed_of(dev: Dict[str, Any], speeds: Dict[str, Dict[str, Any]]) -> Optional[float]:
    rec = speeds.get(device_key(dev))
    return float(rec["stepsPerSecond"]) if rec else None


def _scene_spec():
    from .studio.benchmark import BenchmarkSpec

    return BenchmarkSpec(width=800, height=600, views=16, splats=200_000, steps=STEPS, measure_from=100)


def config() -> Dict[str, Any]:
    """The fixed recipe: the benchmark's settings, splats held at the scene's 200 k (no densification)."""
    spec = _scene_spec()
    cfg = spec.training_config()
    cfg.update(iterations=STEPS, densifyStart=STEPS + 1, densifyStop=STEPS + 2, maxGaussians=spec.splats)
    return cfg


def measure(dev: Dict[str, Any], scene_root: Path, binary: Optional[str] = None, timeout: float = 1800.0,
            progress: Optional[Callable[[str], None]] = None) -> Dict[str, Any]:
    """Train the speed scene on Vulkan device ``dev`` (a ``--list-devices`` entry); returns its record."""
    from .studio.benchmark import build_scene
    from .vktrain import binary_path

    binary = binary or binary_path()
    if not binary:
        raise RuntimeError("the Vulkan trainer (splatwork-vk-train) is not installed")
    say = progress or (lambda m: None)
    say("building the test scene")
    scene = build_scene(Path(scene_root), _scene_spec())
    say(f"training {STEPS} steps on {dev.get('name')}")
    with tempfile.TemporaryDirectory(prefix="splatwork-gpuspeed-") as tmp:
        cfg_file = Path(tmp) / "config.json"
        cfg_file.write_text(json.dumps(config()), encoding="utf-8")
        argv = [binary, "train", "--dataset", str(scene), "--config", str(cfg_file), "--out", str(Path(tmp) / "out.ply"),
                "--device", str(int(dev["index"]))]
        t0 = time.time()
        r = subprocess.run(argv, capture_output=True, text=True, timeout=timeout, stdin=subprocess.DEVNULL)
        wall = time.time() - t0
    done = err = None
    for line in r.stdout.splitlines():
        if not line.startswith("{"):
            continue
        try:
            m = json.loads(line)
        except ValueError:
            continue
        if m.get("type") == "done":
            done = m
        elif m.get("type") == "error":
            err = m.get("message")
    if r.returncode != 0 or not done or not done.get("stepsPerSecond"):
        tail = (err or (r.stderr or "").strip().splitlines()[-1:] or [f"exit code {r.returncode}"])
        raise RuntimeError(f"{dev.get('name')}: {tail if isinstance(tail, str) else tail[0]}")
    spec = _scene_spec()
    return {"name": dev.get("name"), "driver": dev.get("driver"), "pci": dev.get("pci"),
            "deviceType": dev.get("deviceType"), "memoryMB": dev.get("memoryMB"),
            "stepsPerSecond": float(done["stepsPerSecond"]), "msPerStep": done.get("msPerStep"),
            "splats": spec.splats, "megapixels": spec.megapixels, "steps": STEPS, "wallSeconds": round(wall, 1),
            "date": datetime.now(timezone.utc).isoformat(timespec="seconds")}


def measure_all(devices: List[Dict[str, Any]], scene_root: Path, path: Optional[Path] = None,
                progress: Optional[Callable[[str], None]] = None) -> Dict[str, Dict[str, Any]]:
    """Measure each device in turn and save as each finishes; a device that fails is reported, not saved."""
    out: Dict[str, Dict[str, Any]] = {}
    for dev in devices:
        try:
            rec = measure(dev, scene_root, progress=progress)
        except (RuntimeError, OSError, subprocess.SubprocessError) as e:
            out[device_key(dev)] = {"name": dev.get("name"), "error": str(e)}
            continue
        out[device_key(dev)] = rec
        save({device_key(dev): rec}, path)
    return out
