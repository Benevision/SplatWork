"""Measured training speed of each Vulkan GPU, for the job slots (slots.py).

Instead of guessing from spec sheets, each device trains the standard benchmark's scene (studio/benchmark.py: the
Mac's system benchmark — 48 photos of 1600×1200, 1 M splats, held fixed) through ``splatwork-vk-train --device N``
under full load: after the first step, WARMUP seconds of warm-up, then the steps done in a WINDOW-second window
(timed here from the program's status lines) give steps/s; then the run is cancelled. A device the full scene
doesn't fit falls back to a smaller one (SIZES); the record says which, and speeds are only compared at the same
size. The GPU memory in use above the idle level is sampled for ``peakMemoryMB`` (sysfs for AMD / Intel — VRAM +
GTT —, NVML for NVIDIA). Run on demand — *Settings → GPUs → Measure GPU speed*, or ``splatworker measure-gpus`` —
while the GPUs are idle, cached per device and driver in ``gpu-speeds.json`` in the app's config dir (a driver
update means measuring again; records of an older RECIPE count as unmeasured). The server re-reads the file
whenever it changes: no restart needed.
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
RECIPE = 2                            # 1 (1.7.3/1.7.4): 300 steps of a tiny scene — too light for a big GPU
WARMUP = 10.0                         # seconds after the first step
WINDOW = 35.0                         # timed seconds
MIN_STEPS = 3                         # a very slow device: the window grows until this many steps (to MAX_RUN)
MAX_RUN = 110.0                       # seconds per size, start-up included
# (name, width, height, photos, splats): the standard benchmark scene first, then smaller ones that fit an iGPU
SIZES = [("standard", 1600, 1200, 48, 1_000_000), ("medium", 800, 600, 16, 200_000)]


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
            data = {k: v for k, v in data.items()
                    if isinstance(v, dict) and v.get("stepsPerSecond") and v.get("recipe") == RECIPE}
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


def record_of(dev: Optional[Dict[str, Any]], speeds: Dict[str, Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    return speeds.get(device_key(dev)) if dev else None


def size_rank(size: Optional[str]) -> int:
    """0 for the largest scene; a device that only fit a smaller scene ranks higher."""
    names = [z[0] for z in SIZES]
    return names.index(size) if size in names else len(names)


def scene_spec(size: str):
    from .studio.benchmark import BenchmarkSpec

    _, w, h, n, splats = next(z for z in SIZES if z[0] == size)
    return BenchmarkSpec(width=w, height=h, views=n, splats=splats)


def scene_dir(cache_dir: Path, size: str) -> Path:
    """The standard size shares `splatworker benchmark`'s cached scene."""
    from .studio.benchmark import SCENE_DIR

    return Path(cache_dir) / (SCENE_DIR if size == "standard" else f"gpu-speed-scene-{size}")


def config(size: str = "standard") -> Dict[str, Any]:
    """The benchmark's settings with the splats held at the scene's count (no densification) and no end in sight:
    the run is cancelled once the window is timed."""
    spec = scene_spec(size)
    cfg = spec.training_config()
    cfg.update(iterations=10_000_000, densifyStart=10_000_000, densifyStop=10_000_001, maxGaussians=spec.splats,
               shDegreeInterval=1)       # full SH degree from the start: the steady-state load
    return cfg


def _memory_used_mb(dev: Dict[str, Any]) -> Optional[int]:
    """GPU memory in use now: sysfs VRAM + GTT (AMD, Intel; an iGPU works mostly from GTT), else NVML."""
    pci = dev.get("pci")
    if pci:
        base = Path("/sys/bus/pci/devices") / pci
        total = 0
        found = False
        for f in ("mem_info_vram_used", "mem_info_gtt_used"):
            try:
                total += int((base / f).read_text().strip())
                found = True
            except (OSError, ValueError):
                pass
        if found:
            return total // 2**20
        try:
            from .gpu import list_gpus

            for g in list_gpus():
                if g.get("pci") == pci:
                    return int(g["memoryMB"]) - int(g["freeMemoryMB"])
        except Exception:  # noqa: BLE001
            pass
    return None


class Unfit(RuntimeError):
    """The scene didn't fit the device (or the run failed): try a smaller one."""


def time_run(argv: List[str], memory: Callable[[], Optional[int]] = lambda: None,
             tick: Callable[[str, float], None] = lambda phase, secs: None,
             warmup: Optional[float] = None, window: Optional[float] = None,
             clock: Callable[[], float] = time.monotonic) -> Dict[str, Any]:
    """Run the program, time the window from its status lines, cancel it. Returns {stepsPerSecond, windowSeconds,
    windowSteps, peakMemoryMB}. ``tick(phase, seconds)``: "starting" | "warming up" | "measuring"."""
    warmup = WARMUP if warmup is None else warmup
    window = WINDOW if window is None else window
    proc = subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                            bufsize=1)
    watchdog = threading.Timer(MAX_RUN + 60.0, proc.kill)    # a device that hangs without printing anything
    watchdog.daemon = True
    watchdog.start()
    t0 = clock()
    base = memory()
    peak = 0
    first = start = None          # (time, step) of the first step, of the window's start
    last = None
    err = None
    stderr_tail: List[str] = []
    threading.Thread(target=lambda: stderr_tail.extend(proc.stderr.read().splitlines()[-5:]), daemon=True).start()
    try:
        for line in proc.stdout:
            now = clock()
            if line.startswith("{"):
                try:
                    m = json.loads(line)
                except ValueError:
                    m = {}
                if m.get("type") == "error":
                    err = m.get("message")
                step = m.get("step") if m.get("type") == "status" else None
                if isinstance(step, int) and step > 0:
                    if first is None:
                        first = (now, step)
                    elif start is None and now - first[0] >= warmup:
                        start = (now, step)
                    elif start is not None:
                        last = (now, step)
            used = memory()
            if used is not None and base is not None:
                peak = max(peak, used - base)
            if first is None:
                tick("starting", now - t0)
            elif start is None:
                tick("warming up", now - first[0])
            else:
                tick("measuring", now - start[0])
            if start is not None and last is not None and (
                    (now - start[0] >= window and last[1] - start[1] >= MIN_STEPS) or now - t0 >= MAX_RUN):
                break
            if now - t0 >= MAX_RUN and start is None:
                break
    finally:
        if proc.poll() is None:
            try:
                proc.stdin.write("cancel\n")
                proc.stdin.flush()
            except OSError:
                pass
            try:
                proc.wait(30)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(10)
        watchdog.cancel()
    if start is None or last is None or last[1] <= start[1]:
        why = err or (stderr_tail[-1] if stderr_tail else f"exit code {proc.returncode}")
        raise Unfit(why if (first is None or err) else "too slow to time at this size")
    secs = last[0] - start[0]
    return {"stepsPerSecond": (last[1] - start[1]) / secs, "windowSeconds": round(secs, 1),
            "windowSteps": last[1] - start[1], "peakMemoryMB": peak if base is not None else None}


def measure(dev: Dict[str, Any], cache_dir: Path, binary: Optional[str] = None,
            progress: Optional[Callable[[str], None]] = None, sizes: Optional[List[str]] = None) -> Dict[str, Any]:
    """Time device ``dev`` (a ``--list-devices`` entry) at the largest size that fits; returns its record."""
    from .studio.benchmark import build_scene
    from .vktrain import binary_path

    binary = binary or binary_path()
    if not binary:
        raise RuntimeError("the Vulkan trainer (splatwork-vk-train) is not installed")
    say = progress or (lambda m: None)
    label = dev.get("label") or dev.get("name")
    failures = []
    for size in sizes or [z[0] for z in SIZES]:
        spec = scene_spec(size)
        say(f"{label}: building the {size} test scene")
        scene = build_scene(scene_dir(cache_dir, size), spec)
        with tempfile.TemporaryDirectory(prefix="splatwork-gpuspeed-") as tmp:
            cfg_file = Path(tmp) / "config.json"
            cfg_file.write_text(json.dumps(config(size)), encoding="utf-8")
            argv = [binary, "train", "--dataset", str(scene), "--config", str(cfg_file),
                    "--out", str(Path(tmp) / "out.ply"), "--device", str(int(dev["index"]))]
            last_said = [None]

            def tick(phase, secs):
                if (phase, int(secs)) != last_said[0]:
                    last_said[0] = (phase, int(secs))
                    say(f"Measuring {label} ({size} scene): {phase}, {int(secs)} s")
            try:
                r = time_run(argv, memory=lambda: _memory_used_mb(dev), tick=tick)
            except Unfit as e:
                failures.append(f"{size}: {e}")
                continue
        return {"name": dev.get("name"), "label": label, "driver": dev.get("driver"), "pci": dev.get("pci"),
                "deviceType": dev.get("deviceType"), "memoryMB": dev.get("memoryMB"), "recipe": RECIPE,
                "size": size, "splats": spec.splats, "photos": spec.views, "width": spec.width, "height": spec.height,
                **r, "date": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                **({"failedSizes": failures} if failures else {})}
    raise RuntimeError(f"{label}: " + "; ".join(failures))


def measure_all(devices: List[Dict[str, Any]], cache_dir: Path, path: Optional[Path] = None,
                progress: Optional[Callable[[str], None]] = None) -> Dict[str, Dict[str, Any]]:
    """Measure each device in turn and save as each finishes; a device that fails is reported, not saved."""
    out: Dict[str, Dict[str, Any]] = {}
    for dev in devices:
        try:
            rec = measure(dev, cache_dir, progress=progress)
        except (RuntimeError, OSError, subprocess.SubprocessError) as e:
            out[device_key(dev)] = {"name": dev.get("name"), "error": str(e)}
            continue
        out[device_key(dev)] = rec
        save({device_key(dev): rec}, path)
    return out


def comparable(records: List[Dict[str, Any]]) -> bool:
    """Speeds compare only when measured at the same scene size."""
    return len({r.get("size") for r in records if r.get("stepsPerSecond")}) <= 1
