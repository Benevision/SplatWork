"""Training benchmark of this PC (Mac: ``Remote/SystemBenchmark.swift``, ``BenchmarkStore``).

The standard benchmark trains a synthetic scene — textured ground, boxes and pillars seen by 48 cameras on two
rings, 1600×1200 photos, seeded with 1 M points so training runs at full load from the first step — for 2,000
steps with fixed settings, and measures steps per second between step 1,000 and the end. The Mac runs the same
recipe on itself and on every worker, so the numbers are comparable (the photos here are rendered by a simple
numpy point splatter instead of the Mac's trainer: the content differs a little, the load does not).

Results are stored in ``benchmarks.json`` in the app's config dir with the Mac's ``DeviceBenchmark`` fields
(``{deviceId: {deviceId, name, date, gpu, software, trainStepsPerSecond, splats, megapixels, uploadMBps,
downloadMBps}}``); :func:`load_benchmarks` / :func:`throughput` are the API for schedulers.

The job is an ordinary training job of a worker: the Studio registers the scene folder with its own server
(``/v1/datasets/local``), ``splatworker benchmark`` uploads it as a zip (any worker, also timing the upload).
"""
from __future__ import annotations

import json
import math
import os
import shutil
import time
import zipfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np
from .. import PRODUCT_NAME

FILE = "benchmarks.json"
SCENE_DIR = "benchmark-scene-v1"


@dataclass
class BenchmarkSpec:
    width: int = 1600
    height: int = 1200
    views: int = 48
    splats: int = 1_000_000
    steps: int = 2000
    measure_from: int = 1000

    @property
    def megapixels(self) -> float:
        return self.width * self.height / 1e6

    def training_config(self) -> Dict[str, Any]:
        """The Mac's ``SystemBenchmark.config`` in worker keys (PROTOCOL.md §3)."""
        return {"iterations": self.steps, "maxGaussians": self.splats, "maxImageSize": max(self.width, self.height),
                "densifyStart": min(200, self.steps), "densifyStop": max(1, int(self.steps * 0.75)),
                "shDegree": 3, "shDegreeInterval": max(1, self.steps * 3 // 20),
                "evalInterval": 0, "previewInterval": 0, "appearanceCorrection": False, "transientMasking": False,
                "depthWeight": 0.0, "skybox": "off", "seed": 7}


class BenchmarkCancelled(Exception):
    pass


class BenchmarkError(Exception):
    pass


# ---------------------------------------------------------------------------------------------- results store
def load_benchmarks(config_dir: Path) -> Dict[str, Dict[str, Any]]:
    """Saved results ``{deviceId: record}`` (Mac ``DeviceBenchmark`` fields); {} when none or unreadable."""
    try:
        data = json.loads((Path(config_dir) / FILE).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if not isinstance(data, dict):
        return {}
    return {k: v for k, v in data.items()
            if isinstance(v, dict) and isinstance(v.get("trainStepsPerSecond"), (int, float))}


def save_benchmark(config_dir: Path, record: Dict[str, Any]) -> None:
    config_dir = Path(config_dir)
    config_dir.mkdir(parents=True, exist_ok=True)
    data = load_benchmarks(config_dir)
    data[record["deviceId"]] = record
    tmp = config_dir / (FILE + ".tmp")
    tmp.write_text(json.dumps(data, indent=1, sort_keys=True), encoding="utf-8")
    os.replace(tmp, config_dir / FILE)


def sorted_results(config_dir: Path) -> List[Dict[str, Any]]:
    return sorted(load_benchmarks(config_dir).values(), key=lambda r: -float(r["trainStepsPerSecond"]))


def throughput(record: Dict[str, Any]) -> float:
    """Scheduler throughput (1.0 ≈ 20 benchmark steps/s), as the Mac's ``BenchmarkStore.throughput``."""
    return float(record["trainStepsPerSecond"]) / 20.0


def seconds_per_step(record: Dict[str, Any], splats: int, megapixels: float) -> float:
    """Per-step cost for a job of this size, scaled from the benchmark load (half splat-, half pixel-bound)."""
    base = 1.0 / max(float(record["trainStepsPerSecond"]), 0.01)
    s = 0.5 * max(splats, 1) / max(int(record.get("splats") or 1), 1) + \
        0.5 * max(megapixels, 0.05) / max(float(record.get("megapixels") or 0.05), 0.05)
    return base * s


# ---------------------------------------------------------------------------------------------- the scene
def _truth(rng: np.random.Generator):
    """~420 k ground-truth points (position, colour 0–1): noise-textured ground, boxes, pillars (down is +y)."""
    def noise(x, z):
        return (np.sin(x * 3.1) * np.cos(z * 2.7) + np.sin(x * 11.3 + z * 7.1) * 0.5
                + np.sin(x * 31 - z * 23) * 0.25) / 1.75 * 0.5 + 0.5
    n = 300_000
    x, z = rng.uniform(-6, 6, n), rng.uniform(-6, 6, n)
    v = noise(x, z)
    chk = np.where((np.floor(x) + np.floor(z)) % 2 == 0, 0.1, -0.1)
    pos = [np.stack([x, np.ones(n), z], 1)]
    col = [np.stack([0.35 + 0.4 * v + chk, 0.45 + 0.3 * v, 0.25 + 0.2 * v], 1)]
    for b in range(24):
        cx, cz = rng.uniform(-4.5, 4.5, 2)
        hs = np.array([rng.uniform(0.2, 0.7), rng.uniform(0.2, 1.2), rng.uniform(0.2, 0.7)])
        hue = np.array([1 if b % 3 == 0 else 0.3, 0.9 if b % 3 == 1 else 0.35, 1 if b % 3 == 2 else 0.4])
        p = rng.uniform(-1, 1, (4000, 3))
        axis = rng.integers(0, 3, 4000)
        r = np.arange(4000)
        p[r, axis] = np.where(p[r, axis] < 0, -1.0, 1.0)
        p[(axis == 1) & (p[:, 1] > 0), 1] = -1.0           # no bottom face
        q = np.array([cx, 1 - hs[1], cz]) + p * hs
        shade = 0.55 + 0.45 * noise(q[:, 0] * 4 + q[:, 1] * 3, q[:, 2] * 4 - q[:, 1])
        pos.append(q)
        col.append(hue[None] * shade[:, None])
    for _ in range(12):
        cx, cz = rng.uniform(-5, 5, 2)
        k = np.arange(2000)
        a = rng.uniform(0, 2 * math.pi, 2000)
        pos.append(np.stack([cx + 0.06 * np.cos(a), 1 - k / 2000 * 2.5, cz + 0.06 * np.sin(a)], 1))
        col.append(np.array([0.9, 0.9, 0.85])[None] * (0.7 + 0.3 * (k % 50) / 50)[:, None])
    return np.concatenate(pos).astype(np.float32), np.clip(np.concatenate(col), 0, 1).astype(np.float32)


def _cameras(spec: BenchmarkSpec) -> List[Dict[str, Any]]:
    fx = 1400.0 * spec.width / 1600
    target = np.array([0, 0.3, 0])
    out = []
    for i in range(spec.views):
        ring, a = i % 2, i / spec.views * 2 * math.pi
        r, y = (6.0, -2.2) if ring == 0 else (4.5, -3.6)
        eye = np.array([r * math.cos(a), y, r * math.sin(a)])
        zv = target - eye
        zv /= np.linalg.norm(zv)
        xv = np.cross([0, 1.0, 0], zv)
        xv /= np.linalg.norm(xv)
        yv = np.cross(zv, xv)
        m = np.eye(4)
        m[:3, 0], m[:3, 1], m[:3, 2], m[:3, 3] = xv, yv, zv, eye
        out.append({"id": i, "image": f"images/{i:03d}.jpg", "width": spec.width, "height": spec.height,
                    "fx": fx, "fy": fx, "cx": spec.width / 2, "cy": spec.height / 2,
                    "cameraToWorld": m.tolist(), "holdout": i % 8 == 7})
    return out


def _render(pos, col, cam, down: int = 4) -> np.ndarray:
    """Nearest-point z-buffer at 1/``down`` resolution (each point ≈ a pixel there), upscaled: a cheap photo."""
    from PIL import Image

    w, h = cam["width"] // down, cam["height"] // down
    c2w = np.asarray(cam["cameraToWorld"])
    R, t = c2w[:3, :3], c2w[:3, 3]
    pc = (pos - t) @ R
    z = pc[:, 2]
    ok = z > 0.1
    u = (cam["fx"] * pc[:, 0] / np.where(ok, z, 1) + cam["cx"]) / down
    v = (cam["fy"] * pc[:, 1] / np.where(ok, z, 1) + cam["cy"]) / down
    ok &= (u >= 0) & (v >= 0) & (u < w) & (v < h)
    idx = np.nonzero(ok)[0]
    idx = idx[np.argsort(-z[idx], kind="stable")]           # far → near: nearer points written last win
    img = np.empty((h, w, 3), np.float32)
    img[:] = np.linspace([0.55, 0.7, 0.9], [0.8, 0.85, 0.9], h)[:, None, :]      # sky gradient
    zb = np.full((h, w), np.inf, np.float32)
    vi, ui = v[idx].astype(np.int64), u[idx].astype(np.int64)
    img[vi, ui] = col[idx]
    zb[vi, ui] = z[idx]
    # Close the gaps between points: a pixel that is empty or much farther than a neighbour (seen through a
    # surface) takes that neighbour's colour; a few rounds fill holes up to ~4 px wide.
    for _ in range(4):
        best_z, best_c = zb.copy(), img.copy()
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                zp = np.pad(zb, 1, constant_values=np.inf)       # no wrap-around at the borders
                nz = zp[1 + dy:1 + dy + h, 1 + dx:1 + dx + w]
                nc = np.pad(img, ((1, 1), (1, 1), (0, 0)), mode="edge")[1 + dy:1 + dy + h, 1 + dx:1 + dx + w]
                take = nz * 1.15 < best_z
                best_z = np.where(take, nz, best_z)
                best_c = np.where(take[..., None], nc, best_c)
        fill = best_z * 1.15 < zb
        zb = np.where(fill, best_z, zb)
        img = np.where(fill[..., None], best_c, img)
    im = Image.fromarray((img * 255).astype(np.uint8)).resize((cam["width"], cam["height"]), Image.BILINEAR)
    return np.asarray(im)


def _write_points(path: Path, xyz: np.ndarray, rgb: np.ndarray) -> None:
    n = len(xyz)
    rec = np.zeros(n, dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("red", "u1"), ("green", "u1"),
                             ("blue", "u1")])
    rec["x"], rec["y"], rec["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    c = np.clip(rgb * 255, 0, 255).astype(np.uint8)
    rec["red"], rec["green"], rec["blue"] = c[:, 0], c[:, 1], c[:, 2]
    head = ("ply\nformat binary_little_endian 1.0\n"
            f"element vertex {n}\nproperty float x\nproperty float y\nproperty float z\n"
            "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
    with open(path, "wb") as fh:
        fh.write(head.encode())
        fh.write(rec.tobytes())


def build_scene(root: Path, spec: BenchmarkSpec = BenchmarkSpec(),
                progress: Optional[Callable[[str, float], None]] = None,
                cancelled: Optional[Callable[[], bool]] = None) -> Path:
    """Create (or reuse) the benchmark dataset in ``root``. Photos are rendered and written one at a time."""
    root = Path(root)
    stamp = root / "spec.json"
    want = asdict(spec)
    try:
        if json.loads(stamp.read_text()) == want and (root / "dataset.json").is_file():
            doc = json.loads((root / "dataset.json").read_text())
            if all((root / c["image"]).is_file() for c in doc["cameras"]):
                return root
    except (OSError, ValueError, KeyError):
        pass
    shutil.rmtree(root, ignore_errors=True)
    (root / "images").mkdir(parents=True)
    say = progress or (lambda m, f: None)
    say("Building the benchmark scene", 0.0)
    rng = np.random.default_rng(7)
    pos, col = _truth(rng)
    cams = _cameras(spec)
    from PIL import Image

    for i, c in enumerate(cams):
        if cancelled is not None and cancelled():
            raise BenchmarkCancelled()
        Image.fromarray(_render(pos, col, c)).save(root / c["image"], quality=92)
        say(f"Rendering the benchmark photos ({i + 1}/{len(cams)})", (i + 1) / len(cams))
    # Starting points: the truth's centres, jittered, repeated up to the splat budget.
    k = np.arange(spec.splats) % len(pos)
    xyz = pos[k] + rng.uniform(-0.02, 0.02, (spec.splats, 3)).astype(np.float32)
    _write_points(root / "points3d.ply", xyz, col[k])
    doc = {"format": "splatdataset", "version": 1, "convention": "opencv", "upVector": [0, -1, 0],
           "sceneCenter": [0, 0, 0], "sceneExtent": 7.0, "points": "points3d.ply", "cameras": cams}
    (root / "dataset.json").write_text(json.dumps(doc), encoding="utf-8")
    stamp.write_text(json.dumps(want), encoding="utf-8")
    return root


# ---------------------------------------------------------------------------------------------- running
def measure(samples: List[tuple], measure_from: int) -> float:
    """Steps per second from (step, seconds[, seconds…]) samples, in poll order: from the first sample at or past
    ``measure_from`` to the last one (all samples when the run never got there, e.g. very short test runs).
    Each further column is another clock (the worker's ``elapsedSeconds`` or None, then this side's wall clock):
    the first one every used sample has and that advances is taken."""
    s = [x for x in samples if x[0] > 0]
    use = [x for x in s if x[0] >= measure_from] or s
    if len(use) >= 2 and use[-1][0] > use[0][0]:
        for k in range(1, len(use[0])):
            a, b = use[0][k], use[-1][k]
            if a is not None and b is not None and all(x[k] is not None for x in use) and b > a:
                return (use[-1][0] - use[0][0]) / (b - a)
    raise BenchmarkError(f"the benchmark run was too short to measure ({len(s)} progress samples)")


def _upload_zip(client, folder: Path, tmp_dir: Path) -> tuple:
    """Zip the scene and POST it (the normal remote path). Returns (datasetId, bytes, seconds)."""
    tmp_dir.mkdir(parents=True, exist_ok=True)
    z = tmp_dir / f"splat-bench-{os.getpid()}.zip"
    try:
        with zipfile.ZipFile(z, "w", zipfile.ZIP_STORED) as zf:
            for p in sorted(folder.rglob("*")):
                if p.is_file() and p.name != "spec.json":
                    zf.write(p, p.relative_to(folder).as_posix())
        size = z.stat().st_size
        t0 = time.perf_counter()
        with client._req("POST", "/v1/datasets", z.read_bytes(), "application/zip", timeout=600) as resp:
            did = json.loads(resp.read())["datasetId"]
        return did, size, time.perf_counter() - t0
    finally:
        try:
            z.unlink()
        except OSError:
            pass


def run(client, scene_root: Path, spec: BenchmarkSpec = BenchmarkSpec(), local: bool = True,
        progress: Optional[Callable[[str, float], None]] = None, cancelled: Optional[Callable[[], bool]] = None,
        priority: int = 1, poll: float = 1.0, device_id: str = "local", name: Optional[str] = None,
        software: Optional[str] = None, tmp_dir: Optional[Path] = None, slot: Optional[str] = None,
        trainer: Optional[str] = None, extra_config: Optional[Dict[str, Any]] = None,
        keep_dataset: Optional[str] = None) -> Dict[str, Any]:
    """Build the scene, train it on ``client``'s worker and return the result record (not saved).

    ``local``: register the folder with the desktop app's own server (no copy) instead of uploading a zip.
    ``slot``: run on that job slot of the worker (``/v1/info`` "slots" id, worker ≥ 1.7.3); ``trainer``: "gsplat" |
    "vulkan" for this job (worker ≥ 1.7.9); ``extra_config`` overrides the recipe (e.g. strategy, evalInterval);
    ``keep_dataset``: a dataset id already on the worker (not registered or deleted here). The record also has
    the job's ``numGaussians``, ``evalSsim``, ``totalSeconds``, ``peakVramMB`` (GPU memory in use above the start,
    from ``/v1/info``) and its ``log``."""
    say = progress or (lambda m, f: None)
    stop = cancelled or (lambda: False)
    scene = build_scene(scene_root, spec, lambda m, f: say(m, 0.1 * f), stop)
    info = client.info()
    up_mbps = None
    if keep_dataset:
        did = keep_dataset
    elif local:
        did = client.local_dataset(scene)["datasetId"]
    else:
        say("Uploading the benchmark scene", 0.1)
        did, nbytes, secs = _upload_zip(client, scene, Path(tmp_dir or scene.parent))
        up_mbps = nbytes / 1e6 / secs if secs > 0 else None
    job = None
    try:
        body = {"datasetId": did, "config": {**spec.training_config(), **(extra_config or {})},
                "name": f"SplatWork benchmark{' (' + trainer + ')' if trainer else ''}"}
        if local:
            body["priority"] = priority
        if slot:
            body["slot"] = slot
        if trainer:
            body["trainer"] = trainer

        def vram_used() -> Optional[int]:
            try:
                gs = [g for g in client.info().get("gpus") or [] if isinstance(g, dict)]
            except Exception:  # noqa: BLE001
                return None
            used = [int(g["memoryMB"]) - int(g["freeMemoryMB"]) for g in gs
                    if g.get("memoryMB") is not None and g.get("freeMemoryMB") is not None]
            return sum(used) if used else None
        base_vram = vram_used()
        peak_vram = 0
        job = client.post("/v1/jobs", body)["jobId"]
        samples: List[tuple] = []
        t_start = time.perf_counter()
        while True:
            if stop():
                try:
                    client.cancel(job)
                except Exception:  # noqa: BLE001
                    pass
                raise BenchmarkCancelled()
            s = client.job(job)
            if base_vram is not None and s.get("state") in ("loading", "training"):
                v = vram_used()
                if v is not None:
                    peak_vram = max(peak_vram, v - base_vram)
            step = int(s.get("step") or 0)
            if s.get("state") == "training" and step > 0:
                el = s.get("elapsedSeconds")
                samples.append((step, float(el) if isinstance(el, (int, float)) else None,
                                time.perf_counter() - t_start))
            if s.get("state") in ("completed", "failed", "cancelled"):
                break
            where = "this PC" if local else (info.get("name") or "the worker")
            say(f"Training on {where} — step {step:,} / {spec.steps:,}" if step else
                f"Training on {where}: {s.get('message') or s.get('state')}", 0.1 + 0.85 * step / spec.steps)
            time.sleep(poll)
        if s["state"] != "completed":
            raise BenchmarkError(f"benchmark job {s['state']}: {s.get('message') or ''}".strip())
        rate = measure(samples, spec.measure_from)
        down_mbps = None
        if not local:
            say("Downloading the result", 0.96)
            import tempfile

            with tempfile.TemporaryDirectory(dir=tmp_dir) as td:
                t0 = time.perf_counter()
                p = client.download(f"/v1/jobs/{job}/result.ply", Path(td) / "r.ply")
                secs = time.perf_counter() - t0
                down_mbps = p.stat().st_size / 1e6 / secs if secs > 0 else None
        gpus = info.get("gpus") or []
        try:
            log_text = client.log(job)
        except Exception:  # noqa: BLE001
            log_text = ""
        say("Done", 1.0)
        return {"deviceId": device_id, "name": name or info.get("name") or "This PC",
                "date": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
                "gpu": (gpus[0].get("name") if gpus and isinstance(gpus[0], dict) else None),
                "software": software or f"{PRODUCT_NAME} {info.get('workerVersion') or ''} · "
                                        f"{info.get('backend') or ''}".strip(" ·"),
                "trainStepsPerSecond": round(rate, 3), "splats": spec.splats,
                "megapixels": round(spec.megapixels, 2), "uploadMBps": up_mbps, "downloadMBps": down_mbps,
                # quality at the end of the run (a broken GPU backend trains fast but badly)
                "psnr": s.get("psnr"), "evalPsnr": s.get("evalPsnr"), "gpuVendor": (
                    gpus[0].get("vendor") if gpus and isinstance(gpus[0], dict) else None),
                "trainer": s.get("trainer") or trainer, "numGaussians": s.get("numGaussians"),
                "evalSsim": s.get("evalSsim"), "totalSeconds": s.get("elapsedSeconds"),
                "peakVramMB": peak_vram if base_vram is not None else None, "log": log_text}
    finally:
        if job is not None:
            try:
                client.delete(f"/v1/jobs/{job}")
            except Exception:  # noqa: BLE001
                pass
        if not keep_dataset:
            client.delete_dataset(did)


# ------------------------------------------------------------- CUDA vs Vulkan trainer
COMPARE_FILE = "trainer-comparisons.json"
COMPARE_LOGS = "trainer-comparison-logs"
QUICK_SPEC = BenchmarkSpec(width=800, height=600, views=24, splats=200_000, steps=600, measure_from=200)


def compare_trainers(client, scene_root: Path, spec: BenchmarkSpec = BenchmarkSpec(), local: bool = True,
                     strategies=("mcmc",), slot: Optional[str] = None, quick: bool = False,
                     progress: Optional[Callable[[str, float], None]] = None,
                     cancelled: Optional[Callable[[], bool]] = None, tmp_dir: Optional[Path] = None,
                     poll: float = 1.0) -> Dict[str, Any]:
    """Train the benchmark scene on the same GPU with gsplat (CUDA) and then with the Vulkan trainer — same
    recipe, seed and strategy, held-out PSNR / SSIM evaluated at the end by each — as two ordinary worker jobs
    (the queue and pause rules apply). Returns {date, gpu, size, spec, slot, runs: [...], comparisons: [...]}."""
    say = progress or (lambda m, f: None)
    info = client.info()
    avail = info.get("trainers") or []
    missing = [t for t in ("gsplat", "vulkan") if t not in avail]
    if missing:
        raise BenchmarkError(" and ".join(
            {"gsplat": "gsplat (the CUDA trainer: set up the NVIDIA GPU runtime)",
             "vulkan": "the Vulkan trainer"}[t] for t in missing) + " isn't available on this worker")
    if slot is None:      # a GPU both trainers can use: a CUDA GPU with a Vulkan device
        both = [x for x in info.get("slots") or [] if x.get("cuda") is not None and x.get("vulkan") is not None]
        slot = both[0]["id"] if both else None
        if slot is None:
            raise BenchmarkError("no GPU here runs both trainers (a CUDA GPU with a Vulkan driver)")
    gpu = next((x.get("name") for x in info.get("slots") or [] if x.get("id") == slot), None)
    scene = build_scene(scene_root, spec, lambda m, f: say(m, 0.1 * f), cancelled)
    did = client.local_dataset(scene)["datasetId"] if local else _upload_zip(client, scene, Path(tmp_dir or scene.parent))[0]
    runs: List[Dict[str, Any]] = []
    order = [(st, t) for st in strategies for t in ("gsplat", "vulkan")]
    try:
        for i, (strategy, trainer) in enumerate(order):
            def sub(m, f, i=i, trainer=trainer, strategy=strategy):
                say(f"{trainer} ({strategy}): {m}", 0.1 + 0.9 * (i + min(max(f, 0.0), 1.0)) / len(order))
            rec = run(client, scene_root, spec, local=local, progress=sub, cancelled=cancelled, poll=poll,
                      slot=slot, trainer=trainer, keep_dataset=did, tmp_dir=tmp_dir,
                      extra_config={"strategy": strategy, "evalInterval": spec.steps})
            runs.append({"trainer": trainer, "strategy": strategy, "stepsPerSecond": rec["trainStepsPerSecond"],
                         "totalSeconds": rec.get("totalSeconds"), "peakVramMB": rec.get("peakVramMB"),
                         "numGaussians": rec.get("numGaussians"), "evalPsnr": rec.get("evalPsnr"),
                         "evalSsim": rec.get("evalSsim"), "log": rec.get("log", "")})
    finally:
        try:
            client.delete_dataset(did)
        except Exception:  # noqa: BLE001
            pass
    return {"date": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
            "gpu": gpu, "slot": slot, "size": "quick" if quick else "full", "spec": asdict(spec),
            "worker": info.get("workerVersion"), "runs": runs, "comparisons": comparisons(runs)}


def comparisons(runs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Per strategy: Vulkan against gsplat — speed ratio, PSNR / SSIM differences and a one-line summary."""
    out = []
    for strategy in dict.fromkeys(r["strategy"] for r in runs):
        g = next((r for r in runs if r["strategy"] == strategy and r["trainer"] == "gsplat"), None)
        v = next((r for r in runs if r["strategy"] == strategy and r["trainer"] == "vulkan"), None)
        if not g or not v:
            continue
        ratio = v["stepsPerSecond"] / g["stepsPerSecond"] if g.get("stepsPerSecond") else None
        dp = (v["evalPsnr"] - g["evalPsnr"]) if v.get("evalPsnr") is not None and g.get("evalPsnr") is not None else None
        ds = (v["evalSsim"] - g["evalSsim"]) if v.get("evalSsim") is not None and g.get("evalSsim") is not None else None
        parts = [f"{ratio:.2f}× the speed" if ratio is not None else "speed n/a"]
        if dp is not None:
            parts.append(f"{dp:+.2f} dB")
        if ds is not None:
            parts.append(f"SSIM {ds:+.3f}")
        out.append({"strategy": strategy, "speedRatio": ratio, "psnrDiff": dp, "ssimDiff": ds,
                    "summary": f"Vulkan ({strategy}): " + ", ".join(parts)})
    return out


def save_comparison(config_dir: Path, record: Dict[str, Any], keep: int = 20) -> Path:
    """Newest first in trainer-comparisons.json (logs in trainer-comparison-logs/, not in the JSON)."""
    config_dir = Path(config_dir)
    logs = config_dir / COMPARE_LOGS
    logs.mkdir(parents=True, exist_ok=True)
    stamp = record["date"].replace(":", "").replace("-", "")
    slim = dict(record, runs=[])
    for r in record["runs"]:
        name = f"{stamp}-{r['trainer']}-{r['strategy']}.log"
        (logs / name).write_text(r.get("log") or "", encoding="utf-8")
        slim["runs"].append({k: v for k, v in r.items() if k != "log"} | {"logFile": name})
    path = config_dir / COMPARE_FILE
    try:
        old = json.loads(path.read_text(encoding="utf-8"))
        old = old if isinstance(old, list) else []
    except (OSError, ValueError):
        old = []
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps([slim] + old[:keep - 1], indent=1), encoding="utf-8")
    tmp.replace(path)
    return path


def load_comparisons(config_dir: Path) -> List[Dict[str, Any]]:
    try:
        data = json.loads((Path(config_dir) / COMPARE_FILE).read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except (OSError, ValueError):
        return []
