"""Dense points: a dense point cloud for a solved dataset, so training starts from surfaces instead of the sparse
SfM points (option ``dense`` of an sfm request, or ``task: dense`` on a finished dataset).

Two ways, picked by ``method`` ("auto" tries them in this order):

* ``mvs`` — COLMAP multi-view stereo: ``image_undistorter`` sets up a workspace from the dataset's (already
  undistorted, pinhole) cameras, ``patch_match_stereo`` computes a depth map per photo against its neighbours and
  ``stereo_fusion`` keeps the depths that agree across photos. PatchMatch needs COLMAP built with CUDA (the ``colmap``
  CLI or pycolmap; Homebrew's COLMAP on a Mac has no GPU support), so this is the NVIDIA-worker path.
* ``depth`` — fusing depth maps the dataset carries (iPhone LiDAR / stereo depth) or, without them, depth predicted
  per photo (Depth Anything V2 Small through ``transformers``) and fitted to the sparse points. A pixel only becomes a
  point when its depth agrees with the depth maps of at least ``minViews - 1`` neighbouring photos (the multi-view
  support idea of dense reconstruction: one view's guess alone is not trusted). Runs on the CPU (prediction on the
  GPU when there is one).

Either way the points are merged on a voxel grid (mean position and colour per cell) and capped at ``maxPoints``:
the cell grows until the cloud fits, so memory stays bounded while fusing and training init stays manageable.
Held-out photos are not used. Sky (``sky`` masks) and masked-out pixels (``mask``) give no points.

The result replaces ``points3d.ply`` (the sparse cloud is kept as ``points3d_sparse.ply``), or with
``mode: "add"`` is merged with it. ``dense.json`` next to it says what was done.
"""

from __future__ import annotations

import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .errors import SplatPrepError
from .ply import read_points_ply, write_points_ply

ProgressCb = Callable[[float, str], None]

DEFAULTS = {
    "method": "auto",        # auto | mvs | depth
    "maxPoints": 1_000_000,  # cap of the result (the voxel cell grows until it fits)
    "voxelSize": 0.0,        # 0 = from the photos' resolution and depth
    "mode": "replace",       # replace | add (merge with the sparse points)
    "maxImageSize": 1200,    # mvs: longest side PatchMatch works at
    "depthSize": 384,        # depth: longest side the depth maps are fused at
    "minViews": 3,           # depth: photos that must agree on a point (the reference + minViews-1 neighbours)
    "neighbours": 6,         # depth: neighbours each photo is checked against
}
# a COLMAP MVS workspace needs this many sparse points (per-photo depth ranges and neighbour choice come from them)
MVS_MIN_POINTS = 500
# fitting a predicted depth map needs this many sparse points seen by the photo
FIT_MIN_POINTS = 30
DEPTH_MODEL = "depth-anything/Depth-Anything-V2-Small-hf"
# memory for the depth maps held while fusing
DEPTH_MAP_BUDGET_BYTES = 768 << 20
# COLMAP PatchMatch / fusion cache: this share of the computer's memory (their default is 32 GB)
MVS_CACHE_SHARE = 0.25


def options(raw) -> dict:
    """The dense options of a request (``true`` = defaults), validated. None = off."""
    if raw is None or raw is False:
        return None
    o = dict(DEFAULTS)
    if isinstance(raw, dict):
        if raw.get("enabled") is False:
            return None
        o.update({k: v for k, v in raw.items() if v is not None and k in DEFAULTS})
    elif raw is not True:
        raise SplatPrepError("dense must be true or an object")
    if o["method"] not in ("auto", "mvs", "depth"):
        raise SplatPrepError(f"unknown dense method: {o['method']!r} (expected auto | mvs | depth)")
    if o["mode"] not in ("replace", "add"):
        raise SplatPrepError(f"unknown dense mode: {o['mode']!r} (expected replace | add)")
    o["maxPoints"] = int(min(max(int(o["maxPoints"]), 10_000), 10_000_000))
    o["maxImageSize"] = int(min(max(int(o["maxImageSize"]), 320), 4000))
    o["depthSize"] = int(min(max(int(o["depthSize"]), 64), 1024))
    o["minViews"] = int(min(max(int(o["minViews"]), 2), 8))
    o["neighbours"] = int(min(max(int(o["neighbours"]), o["minViews"] - 1), 16))
    o["voxelSize"] = float(max(float(o["voxelSize"]), 0.0))
    return o


# ------------------------------------------------------------------------------------------ availability


def _total_memory() -> int:
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (ValueError, OSError, AttributeError):
        return 16 << 30


_CUDA_CACHE: Dict[str, bool] = {}


def colmap_cli_cuda() -> Optional[str]:
    """The ``colmap`` executable when it was built with CUDA (``colmap -h`` says "with CUDA"), else None."""
    from .mapper import find_executable

    exe = find_executable("colmap")
    if not exe:
        return None
    if exe not in _CUDA_CACHE:
        try:
            out = subprocess.run([exe, "-h"], capture_output=True, text=True, timeout=30)
            head = (out.stdout + out.stderr)[:400]
            _CUDA_CACHE[exe] = bool(re.search(r"with CUDA", head, re.I))
        except (OSError, subprocess.SubprocessError):
            _CUDA_CACHE[exe] = False
    return exe if _CUDA_CACHE[exe] else None


def pycolmap_cuda() -> bool:
    """pycolmap built with CUDA (checked in a child interpreter: pycolmap and torch don't share a process)."""
    if "pycolmap" not in _CUDA_CACHE:
        from .device import module_installed

        ok = False
        if module_installed("pycolmap"):
            try:
                out = subprocess.run([sys.executable, "-c", "import pycolmap; print(bool(getattr(pycolmap, 'has_cuda', False)))"],
                                     capture_output=True, text=True, timeout=60)
                ok = out.stdout.strip().endswith("True")
            except (OSError, subprocess.SubprocessError):
                ok = False
        _CUDA_CACHE["pycolmap"] = ok
    return _CUDA_CACHE["pycolmap"]


def depth_model_available() -> bool:
    from .device import has_module

    return has_module("torch") and has_module("transformers")


def available_methods() -> dict:
    """What this computer can do: {"mvs": bool, "mvsTool": "colmap"|"pycolmap"|None, "depth": True,
    "predictedDepth": bool}. ``depth`` with sensor depth always works (numpy only)."""
    tool = "colmap" if colmap_cli_cuda() else ("pycolmap" if pycolmap_cuda() else None)
    return {"mvs": tool is not None, "mvsTool": tool, "depth": True, "predictedDepth": depth_model_available()}


# ------------------------------------------------------------------------------------------ dataset


@dataclass
class Cam:
    index: int
    image: Path
    width: int
    height: int
    K: np.ndarray                     # 3x3, image pixels
    c2w: np.ndarray                   # 4x4 OpenCV
    depth: Optional[Path] = None
    depth_size: Optional[Tuple[int, int]] = None
    mask: Optional[Path] = None
    sky: Optional[Path] = None
    name: str = ""

    @property
    def w2c(self) -> np.ndarray:
        return np.linalg.inv(self.c2w)

    @property
    def center(self) -> np.ndarray:
        return self.c2w[:3, 3]

    @property
    def axis(self) -> np.ndarray:
        return self.c2w[:3, 2]


def load_cameras(dataset_dir: Path, image_root: Optional[Path] = None,
                 mask_dirs: Sequence[Tuple[str, Path]] = ()) -> List[Cam]:
    """The dataset's training cameras (held-out ones left out). ``image_root``: where images are found by name when
    the dataset has no copy of them (an sfm export with ``exportImages: false``). ``mask_dirs``: extra
    ("mask"|"sky", directory) to find masks by image stem when the cameras don't name one."""
    ds = json.loads((Path(dataset_dir) / "dataset.json").read_text(encoding="utf-8"))
    cams = []
    for k, c in enumerate(ds.get("cameras", [])):
        if c.get("holdout"):
            continue
        rel = str(c["image"])
        if ".." in Path(rel).parts:
            raise SplatPrepError(f"path escapes the dataset folder: {rel}")
        img = Path(dataset_dir) / rel
        if not img.is_file() and image_root is not None:
            inner = Path(*Path(rel).parts[1:]) if Path(rel).parts[:1] == ("images",) and len(Path(rel).parts) > 1 else Path(rel)
            img = next((p for p in (Path(image_root) / inner, Path(image_root) / Path(rel).name) if p.is_file()), img)
        if not img.is_file():
            continue
        K = np.array([[c["fx"], 0, c["cx"]], [0, c["fy"], c["cy"]], [0, 0, 1]], float)
        cam = Cam(k, img, int(c["width"]), int(c["height"]), K, np.asarray(c["cameraToWorld"], float).reshape(4, 4),
                  name=Path(rel).name)
        if c.get("depth") and c.get("depthWidth") and c.get("depthHeight"):
            p = Path(dataset_dir) / c["depth"]
            if p.is_file() and ".." not in Path(c["depth"]).parts:
                cam.depth, cam.depth_size = p, (int(c["depthWidth"]), int(c["depthHeight"]))
        for key in ("mask", "sky"):
            if c.get(key) and ".." not in Path(c[key]).parts and (Path(dataset_dir) / c[key]).is_file():
                setattr(cam, key, Path(dataset_dir) / c[key])
        stem = Path(rel).stem
        for key, d in mask_dirs:
            if getattr(cam, key) is None:
                for cand in (Path(d) / f"{stem}.png", Path(d) / f"{Path(rel).name}.png"):
                    if cand.is_file():
                        setattr(cam, key, cand)
                        break
        cams.append(cam)
    return cams


def _read_gray(path: Optional[Path], size: Tuple[int, int]) -> Optional[np.ndarray]:
    if path is None:
        return None
    from PIL import Image

    try:
        with Image.open(path) as im:
            return np.asarray(im.convert("L").resize(size, Image.NEAREST))
    except OSError:
        return None


def _read_rgb(path: Path, size: Tuple[int, int]) -> Optional[np.ndarray]:
    from PIL import Image

    from .images import IMAGE_EXTS  # noqa: F401  (registers the HEIC opener)
    try:
        with Image.open(path) as im:
            return np.asarray(im.convert("RGB").resize(size, Image.BILINEAR))
    except OSError:
        return None


def _project(X: np.ndarray, cam: Cam, w: int, h: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """(u, v, z, inside) of world points in ``cam`` at a ``w`` x ``h`` resolution."""
    w2c = cam.w2c
    Xc = X @ w2c[:3, :3].T + w2c[:3, 3]
    z = Xc[:, 2]
    sx, sy = w / cam.width, h / cam.height
    zs = np.where(z > 1e-9, z, 1.0)
    u = (cam.K[0, 0] * Xc[:, 0] / zs + cam.K[0, 2]) * sx
    v = (cam.K[1, 1] * Xc[:, 1] / zs + cam.K[1, 2]) * sy
    inside = (z > 1e-9) & (u >= 0) & (u < w) & (v >= 0) & (v < h)
    return u, v, z, inside


def visible(u: np.ndarray, v: np.ndarray, z: np.ndarray, inside: np.ndarray, w: int, h: int,
            grid: int = 64, slack: float = 1.15) -> np.ndarray:
    """Indices of the projected points that are not hidden behind nearer ones: a coarse depth buffer (``grid``
    cells across) keeps the points within ``slack`` × the nearest depth in their cell. The sparse cloud has points
    seen from everywhere; behind a wall they would drag a depth fit or a photo's depth range."""
    idx = np.nonzero(inside)[0]
    if len(idx) == 0:
        return idx
    gw, gh = grid, max(4, round(grid * h / w))
    cx = np.clip((u[idx] / w * gw).astype(np.int64), 0, gw - 1)
    cy = np.clip((v[idx] / h * gh).astype(np.int64), 0, gh - 1)
    cell = cy * gw + cx
    zmin = np.full(gw * gh, np.inf)
    np.minimum.at(zmin, cell, z[idx])
    return idx[z[idx] <= slack * zmin[cell]]


def median_depths(cams: Sequence[Cam], points: np.ndarray, fallback: float) -> np.ndarray:
    out = np.full(len(cams), fallback)
    if len(points) == 0:
        return out
    pts = points if len(points) <= 50_000 else points[np.random.default_rng(0).choice(len(points), 50_000, replace=False)]
    for i, c in enumerate(cams):
        _, _, z, inside = _project(pts, c, c.width, c.height)
        if inside.sum() >= 5:
            out[i] = float(np.median(z[inside]))
    return out


def choose_neighbours(cams: Sequence[Cam], depths: np.ndarray, k: int) -> List[List[int]]:
    """For each camera, up to ``k`` others that see the same part of the scene from a useful angle: optical axes
    within 75°, and a triangulation angle at the photo's typical depth best between 3° and 20° (too small: no
    parallax to check depth against; too large: the views see different sides)."""
    C = np.stack([c.center for c in cams])
    A = np.stack([c.axis for c in cams])
    out = []
    for i in range(len(cams)):
        cosax = A @ A[i]
        base = np.linalg.norm(C - C[i], axis=1)
        tri = np.degrees(np.arctan2(base, max(depths[i], 1e-9)))
        wt = np.clip(tri / 3.0, 0, 1) * np.clip((40.0 - tri) / 20.0, 0, 1)
        # the point the photo looks at should be in front of the neighbour too
        target = C[i] + A[i] * depths[i]
        front = np.einsum("ij,ij->i", target - C, A) > 0
        score = wt * np.clip(cosax, 0, 1) * front
        score[i] = 0
        score[cosax < math.cos(math.radians(75))] = 0
        order = [j for j in np.argsort(-score) if score[j] > 0][:k]
        out.append([int(j) for j in order])
    return out


# ------------------------------------------------------------------------------------------ voxel grid


class VoxelGrid:
    """Running mean position and colour per voxel. ``add`` merges points in; when the grid holds more than
    ``limit`` cells the cell grows (×1.4) so memory stays bounded."""

    OFF = 1 << 20

    def __init__(self, voxel: float, limit: int):
        self.voxel = float(voxel)
        self.limit = int(limit)
        self.keys = np.zeros(0, np.int64)
        self.sum = np.zeros((0, 3), np.float64)
        self.col = np.zeros((0, 3), np.float64)
        self.n = np.zeros(0, np.float64)

    def _keys(self, xyz: np.ndarray) -> np.ndarray:
        q = np.floor(xyz / self.voxel).astype(np.int64) + self.OFF
        q = np.clip(q, 0, (1 << 21) - 1)
        return (q[:, 0] << 42) | (q[:, 1] << 21) | q[:, 2]

    def _merge(self, keys, s, c, n):
        uk, inv = np.unique(keys, return_inverse=True)
        inv = inv.reshape(-1)
        m = len(uk)
        self.sum = np.stack([np.bincount(inv, s[:, d], m) for d in range(3)], 1)
        self.col = np.stack([np.bincount(inv, c[:, d], m) for d in range(3)], 1)
        self.n = np.bincount(inv, n, m)
        self.keys = uk

    def add(self, xyz: np.ndarray, rgb: np.ndarray) -> None:
        if len(xyz) == 0:
            return
        xyz = np.asarray(xyz, np.float64)
        rgb = np.asarray(rgb, np.float64)
        self._merge(np.concatenate([self.keys, self._keys(xyz)]), np.concatenate([self.sum, xyz]),
                    np.concatenate([self.col, rgb]), np.concatenate([self.n, np.ones(len(xyz))]))
        while len(self.keys) > self.limit:
            self.coarsen(1.4)

    def coarsen(self, factor: float) -> None:
        self.voxel *= factor
        if len(self.keys):
            self._merge(self._keys(self.sum / self.n[:, None]), self.sum, self.col, self.n)

    def fit(self, max_points: int) -> None:
        while len(self.keys) > max_points:
            self.coarsen(1.25)

    def __len__(self) -> int:
        return len(self.keys)

    def points(self) -> Tuple[np.ndarray, np.ndarray]:
        if not len(self.keys):
            return np.zeros((0, 3)), np.zeros((0, 3), np.uint8)
        return self.sum / self.n[:, None], np.clip(np.round(self.col / self.n[:, None]), 0, 255).astype(np.uint8)


def voxel_downsample(xyz: np.ndarray, rgb: np.ndarray, voxel: float, max_points: int) -> Tuple[np.ndarray, np.ndarray, float]:
    """Voxel merge of a point set, the cell grown until at most ``max_points`` remain. Returns (xyz, rgb, voxel)."""
    if len(xyz) == 0:
        return xyz, rgb, voxel
    if voxel <= 0:
        lo, hi = np.percentile(xyz, 1, axis=0), np.percentile(xyz, 99, axis=0)
        voxel = float(max(np.max(hi - lo), 1e-6)) / 4000.0
    g = VoxelGrid(voxel, max(4 * max_points, 1))
    for s in range(0, len(xyz), 2_000_000):
        g.add(xyz[s:s + 2_000_000], rgb[s:s + 2_000_000])
    g.fit(max_points)
    p, c = g.points()
    return p, c, g.voxel


# ------------------------------------------------------------------------------------------ depth maps


def read_sensor_depth(cam: Cam) -> Optional[np.ndarray]:
    w, h = cam.depth_size
    try:
        raw = np.fromfile(cam.depth, dtype="<f2", count=w * h)
    except OSError:
        return None
    if raw.size < w * h:
        return None
    d = raw.reshape(h, w).astype(np.float32)
    d[~np.isfinite(d)] = 0
    d[d < 0] = 0
    return d


def fit_inverse_depth(pred: np.ndarray, z: np.ndarray, iters: int = 8) -> Optional[Tuple[float, float, float]]:
    """Robust fit of 1/z ≈ a·pred + b (relative inverse depth → metric depth). (a, b, median relative error) or None."""
    ok = np.isfinite(pred) & np.isfinite(z) & (z > 0)
    pred, z = pred[ok].astype(np.float64), z[ok].astype(np.float64)
    if pred.size < FIT_MIN_POINTS:
        return None
    y = 1.0 / z
    wt = np.ones_like(y)
    a = b = 0.0
    for _ in range(iters):
        sol, *_ = np.linalg.lstsq(np.stack([pred * wt, wt], 1), y * wt, rcond=None)
        a, b = float(sol[0]), float(sol[1])
        r = np.abs(a * pred + b - y)
        s = 1.4826 * np.median(r) + 1e-12
        wt = 1.0 / np.maximum(1.0, r / (2.0 * s))
    if a <= 0:
        return None
    inv = a * pred + b
    good = inv > 1e-9
    if good.sum() < FIT_MIN_POINTS:
        return None
    rel = np.abs(1.0 / inv[good] - z[good]) / z[good]
    return a, b, float(np.median(rel))


class DepthPredictor:
    """Depth Anything V2 Small (relative inverse depth), loaded on first use."""

    def __init__(self, device: str):
        self.device = device if device in ("cuda", "mps", "xpu") else "cpu"
        self.model = self.proc = None

    def __call__(self, rgb: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
        import torch
        from PIL import Image

        if self.model is None:
            from transformers import AutoImageProcessor, AutoModelForDepthEstimation

            self.proc = AutoImageProcessor.from_pretrained(DEPTH_MODEL)
            self.model = AutoModelForDepthEstimation.from_pretrained(DEPTH_MODEL).to(self.device).eval()
        with torch.no_grad():
            inp = self.proc(images=Image.fromarray(rgb), return_tensors="pt").to(self.device)
            out = self.model(**inp).predicted_depth
            pred = torch.nn.functional.interpolate(out[:, None], size=(size[1], size[0]), mode="bilinear", align_corners=False)
            return pred[0, 0].float().cpu().numpy()

    def close(self) -> None:
        if self.model is not None:
            del self.model
            self.model = None
            try:
                import torch

                if self.device == "cuda":
                    torch.cuda.empty_cache()
            except Exception:  # noqa: BLE001
                pass


def _work_size(cam: Cam, long_side: int) -> Tuple[int, int]:
    s = min(1.0, long_side / max(cam.width, cam.height))
    return max(8, round(cam.width * s)), max(8, round(cam.height * s))


def build_depth_maps(cams: Sequence[Cam], points: np.ndarray, o: dict, device: str, progress: ProgressCb,
                     log: Callable[[str], None]) -> Tuple[Dict[int, np.ndarray], str]:
    """Depth map per camera (float16, 0 = none): the dataset's own (sensor) depth when most cameras have one, else
    predicted and fitted to the sparse points. Returns (maps, source)."""
    n = len(cams)
    long_side = int(min(o["depthSize"], max(64, math.sqrt(DEPTH_MAP_BUDGET_BYTES / max(n, 1) / 2 / 0.75))))
    maps: Dict[int, np.ndarray] = {}
    sensor = sum(c.depth is not None for c in cams) >= 0.5 * n
    if sensor:
        for i, c in enumerate(cams):
            if c.depth is None:
                continue
            d = read_sensor_depth(c)
            if d is None:
                continue
            if max(d.shape) > long_side:
                from PIL import Image

                w, h = _work_size(c, long_side)
                d = np.asarray(Image.fromarray(d).resize((w, h), Image.NEAREST))
            maps[i] = d.astype(np.float16)
            if i % 20 == 0:
                progress(0.4 * (i + 1) / n, f"reading depth {i + 1}/{n}")
        return maps, "sensor"
    if not depth_model_available():
        raise SplatPrepError("no depth maps in the dataset and no depth model installed (PyTorch + transformers)")
    if len(points) < 100:
        raise SplatPrepError("too few sparse points to scale predicted depth")
    pts = points if len(points) <= 200_000 else points[np.random.default_rng(0).choice(len(points), 200_000, replace=False)]
    pred = DepthPredictor(device)
    errs = []
    try:
        for i, c in enumerate(cams):
            w, h = _work_size(c, long_side)
            rgb = _read_rgb(c.image, (w, h))
            if rgb is None:
                continue
            p = pred(rgb, (w, h))
            u, v, z, inside = _project(pts, c, w, h)
            vis = visible(u, v, z, inside, w, h)
            fit = None
            if len(vis) >= FIT_MIN_POINTS:
                samp = p[np.clip(v[vis].astype(np.int64), 0, h - 1), np.clip(u[vis].astype(np.int64), 0, w - 1)]
                fit = fit_inverse_depth(samp, z[vis])
            if fit is None or fit[2] > 0.25:
                continue
            a, b, err = fit
            inv = a * p + b
            d = np.where(inv > 1e-9, 1.0 / np.maximum(inv, 1e-9), 0.0).astype(np.float32)
            d[d > 50 * np.median(z[vis])] = 0       # sky / far noise, not a surface
            maps[i] = d.astype(np.float16)
            errs.append(err)
            if i % 5 == 0:
                progress(0.6 * (i + 1) / n, f"predicting depth {i + 1}/{n}")
    finally:
        pred.close()
    if errs:
        log(f"predicted depth: {len(maps)} of {n} photos fitted to the sparse points "
            f"(median depth error {100 * float(np.median(errs)):.0f} %)")
    return maps, "predicted"


def fuse_depth_maps(cams: Sequence[Cam], maps: Dict[int, np.ndarray], neighbours: List[List[int]], o: dict,
                    tol: float, voxel: float, progress: ProgressCb, start: float = 0.6) -> Tuple[np.ndarray, np.ndarray, float, int]:
    """Back-projects each depth map and keeps the pixels whose depth agrees (within ``tol`` relative) with at least
    ``minViews - 1`` neighbouring depth maps. Returns (xyz, rgb, voxel, raw count)."""
    n = len(cams)
    need = o["minViews"] - 1
    budget = max(2_000, 6 * o["maxPoints"] // max(len(maps), 1))
    grid = VoxelGrid(voxel, 4 * o["maxPoints"])
    rng = np.random.default_rng(0)
    raw = 0
    for i, c in enumerate(cams):
        d = maps.get(i)
        if d is None:
            continue
        nb = [j for j in neighbours[i] if j in maps]
        if len(nb) < need:
            continue
        h, w = d.shape
        d = d.astype(np.float32)
        valid = d > 0
        keep = _read_gray(c.mask, (w, h))
        if keep is not None:
            valid &= keep > 127
        sky = _read_gray(c.sky, (w, h))
        if sky is not None:
            valid &= sky <= 127
        ys, xs = np.nonzero(valid)
        if len(ys) > budget:
            k = rng.choice(len(ys), budget, replace=False)
            ys, xs = ys[k], xs[k]
        if len(ys) == 0:
            continue
        z = d[ys, xs].astype(np.float64)
        sx, sy = w / c.width, h / c.height
        Xc = np.stack(((xs + 0.5 - c.K[0, 2] * sx) / (c.K[0, 0] * sx) * z, (ys + 0.5 - c.K[1, 2] * sy) / (c.K[1, 1] * sy) * z, z), 1)
        X = Xc @ c.c2w[:3, :3].T + c.c2w[:3, 3]
        agree = np.zeros(len(X), np.int32)
        for j in nb:
            dj = maps[j]
            hj, wj = dj.shape
            u, v, zj, inside = _project(X, cams[j], wj, hj)
            ui = np.clip(u.astype(np.int64), 0, wj - 1)
            vi = np.clip(v.astype(np.int64), 0, hj - 1)
            seen = dj[vi, ui].astype(np.float64)
            agree += (inside & (seen > 0) & (np.abs(seen - zj) <= tol * zj)).astype(np.int32)
        ok = agree >= need
        if ok.any():
            rgb = _read_rgb(c.image, (w, h))
            col = rgb[ys[ok], xs[ok]] if rgb is not None else np.full((int(ok.sum()), 3), 128, np.uint8)
            grid.add(X[ok], col)
            raw += int(ok.sum())
        if i % 5 == 0:
            progress(start + (1 - start) * 0.9 * (i + 1) / n, f"fusing depth {i + 1}/{n}: {len(grid):,} points")
    grid.fit(o["maxPoints"])
    xyz, rgb = grid.points()
    return xyz, rgb, grid.voxel, raw


# ------------------------------------------------------------------------------------------ COLMAP MVS


def write_mvs_model(cams: Sequence[Cam], points: np.ndarray, rgb: np.ndarray, out: Path) -> List[int]:
    """A COLMAP model of the dataset's cameras (PINHOLE) with the sparse points as tracks: a point is "seen" by the
    photos it projects into, unless a nearer point covers it there (coarse depth buffer). PatchMatch takes each
    photo's depth range and its source photos from these tracks. Returns the camera indices written."""
    from .colmap_model import Camera, Image as CImage, Point3D, Reconstruction, rotmat_to_qvec

    rec = Reconstruction()
    pts = np.asarray(points, float)
    if len(pts) > 200_000:
        k = np.random.default_rng(0).choice(len(pts), 200_000, replace=False)
        pts, rgb = pts[k], rgb[k]
    obs: Dict[int, List[Tuple[int, int]]] = {}
    used = []
    for i, c in enumerate(cams):
        u, v, z, inside = _project(pts, c, c.width, c.height)
        vis = visible(u, v, z, inside, c.width, c.height)
        if len(vis) < 20:
            continue
        iid = len(used) + 1
        used.append(i)
        rec.cameras[iid] = Camera(iid, "PINHOLE", c.width, c.height, np.array([c.K[0, 0], c.K[1, 1], c.K[0, 2], c.K[1, 2]]))
        w2c = c.w2c
        im = CImage(iid, rotmat_to_qvec(w2c[:3, :3]), w2c[:3, 3].copy(), iid, mvs_name(i, c))
        im.xys = np.stack([u[vis], v[vis]], 1)
        im.point3D_ids = vis.astype(np.int64) + 1
        rec.images[iid] = im
        for k2, p in enumerate(vis):
            obs.setdefault(int(p), []).append((iid, k2))
    for p, tr in obs.items():
        if len(tr) < 2:
            continue
        rec.points3D[p + 1] = Point3D(p + 1, pts[p], np.asarray(rgb[p], np.uint8), 1.0,
                                      np.array([t[0] for t in tr], np.int32), np.array([t[1] for t in tr], np.int32))
    # observations of dropped single-view points must not point at missing points
    for im in rec.images.values():
        im.point3D_ids = np.array([pid if pid in rec.points3D else -1 for pid in im.point3D_ids], np.int64)
    from .colmap_model import write_binary_model

    write_binary_model(rec, out)
    return used


def _mvs_cache_gb() -> int:
    return int(max(2, min(32, _total_memory() * MVS_CACHE_SHARE / (1 << 30))))


def run_colmap_mvs(tool: str, colmap: Optional[str], image_root: Path, model: Path, ws: Path, o: dict,
                   log_path: Optional[Path], progress: ProgressCb, n_images: int) -> Path:
    """undistorter → PatchMatch (geometric consistency) → fusion; returns the fused PLY."""
    from .mapper import pick_option, run_logged

    dense = ws / "dense"
    if dense.exists():
        shutil.rmtree(dense)
    fused = ws / "fused.ply"
    size = o["maxImageSize"]
    cache = _mvs_cache_gb()
    threads = max(1, min(os.cpu_count() or 4, 8))
    seen = {"views": 0}
    total = max(1, 2 * n_images)       # photometric + geometric pass

    def on_line(line: str):
        m = re.search(r"[Pp]rocessing view (\d+)\s*/\s*(\d+)", line)
        if m:
            seen["views"] += 1
            progress(0.05 + 0.8 * min(1.0, seen["views"] / total), f"multi-view stereo: view {m.group(1)}/{m.group(2)}")
        elif "Fusing image" in line:
            m2 = re.search(r"\[(\d+)/(\d+)\]", line)
            if m2:
                progress(0.85 + 0.1 * int(m2.group(1)) / max(1, int(m2.group(2))), f"fusing depth maps {m2.group(1)}/{m2.group(2)}")

    if tool == "colmap":
        progress(0.01, "multi-view stereo: setting up the workspace")
        run_logged([colmap, "image_undistorter", "--image_path", image_root, "--input_path", model,
                    "--output_path", dense, "--output_type", "COLMAP", "--max_image_size", str(size)], log_path)
        pm = [colmap, "patch_match_stereo", "--workspace_path", dense, "--workspace_format", "COLMAP",
              "--PatchMatchStereo.geom_consistency", "true", "--PatchMatchStereo.max_image_size", str(size)]
        if pick_option(colmap, "patch_match_stereo", "PatchMatchStereo.cache_size"):
            pm += ["--PatchMatchStereo.cache_size", str(cache)]
        run_logged(pm, log_path, on_line)
        fu = [colmap, "stereo_fusion", "--workspace_path", dense, "--workspace_format", "COLMAP",
              "--input_type", "geometric", "--output_path", fused, "--StereoFusion.max_image_size", str(size)]
        for opt, val in (("StereoFusion.cache_size", str(cache)), ("StereoFusion.num_threads", str(threads))):
            if pick_option(colmap, "stereo_fusion", opt):
                fu += ["--" + opt, val]
        run_logged(fu, log_path, on_line)
    else:
        run_logged([sys.executable, "-c", _PYCOLMAP_MVS_SCRIPT, str(image_root), str(model), str(dense), str(fused),
                    str(size), str(cache), str(threads)], log_path, on_line)
    if not fused.is_file():
        raise SplatPrepError("multi-view stereo produced no points")
    return fused


_PYCOLMAP_MVS_SCRIPT = r"""
import sys, pycolmap
image_root, model, dense, fused, size, cache, threads = sys.argv[1:8]
size, cache, threads = int(size), int(cache), int(threads)
def setattrs(o, **kw):
    for k, v in kw.items():
        if hasattr(o, k):
            setattr(o, k, v)
uo = pycolmap.UndistortCameraOptions()
setattrs(uo, max_image_size=size)
print("setting up the workspace", flush=True)
pycolmap.undistort_images(dense, model, image_root, undistort_options=uo)
pm = pycolmap.PatchMatchOptions()
setattrs(pm, geom_consistency=True, max_image_size=size, cache_size=cache)
print("patch match stereo", flush=True)
pycolmap.patch_match_stereo(dense, options=pm)
fo = pycolmap.StereoFusionOptions()
setattrs(fo, max_image_size=size, cache_size=cache, num_threads=threads)
print("fusion", flush=True)
pycolmap.stereo_fusion(fused, dense, input_type="geometric", options=fo)
print("done", flush=True)
"""


def drop_sky_points(xyz: np.ndarray, rgb: np.ndarray, cams: Sequence[Cam], max_cams: int = 64) -> Tuple[np.ndarray, np.ndarray, int]:
    """Points that land on the sky (sky masks) in at least half of the photos that see them."""
    with_sky = [c for c in cams if c.sky is not None]
    if not with_sky or len(xyz) == 0:
        return xyz, rgb, 0
    if len(with_sky) > max_cams:
        with_sky = [with_sky[i] for i in np.linspace(0, len(with_sky) - 1, max_cams).round().astype(int)]
    seen = np.zeros(len(xyz), np.int32)
    sky = np.zeros(len(xyz), np.int32)
    for c in with_sky:
        w, h = _work_size(c, 512)
        m = _read_gray(c.sky, (w, h))
        if m is None:
            continue
        u, v, _, inside = _project(xyz, c, w, h)
        idx = np.nonzero(inside)[0]
        seen[idx] += 1
        sky[idx] += (m[v[idx].astype(np.int64).clip(0, h - 1), u[idx].astype(np.int64).clip(0, w - 1)] > 127)
    drop = (seen > 0) & (2 * sky >= seen) & (sky > 0)
    return xyz[~drop], rgb[~drop], int(drop.sum())


# ------------------------------------------------------------------------------------------ entry point


def _sparse(dataset_dir: Path) -> Tuple[Optional[Path], np.ndarray, np.ndarray]:
    ds = json.loads((dataset_dir / "dataset.json").read_text(encoding="utf-8"))
    rel = ds.get("points")
    p = dataset_dir / rel if rel and ".." not in Path(rel).parts else None
    if p is None or not p.is_file():
        return None, np.zeros((0, 3)), np.zeros((0, 3), np.uint8)
    xyz, rgb = read_points_ply(p)
    return p, xyz, rgb


def _auto_voxel(cams: Sequence[Cam], depths: np.ndarray, long_side: int) -> float:
    """About two pixels of the fused depth maps at the photos' typical depth."""
    f = np.array([c.K[0, 0] * min(1.0, long_side / max(c.width, c.height)) for c in cams])
    return float(max(np.median(2.0 * depths / np.maximum(f, 1e-6)), 1e-6))


def densify(dataset_dir: Path, raw_options, progress: Optional[ProgressCb] = None, log: Optional[Callable[[str], None]] = None,
            device: str = "cpu", image_root: Optional[Path] = None, mask_dirs: Sequence[Tuple[str, Path]] = (),
            work_dir: Optional[Path] = None, log_path: Optional[Path] = None) -> dict:
    """Builds the dense cloud of the dataset at ``dataset_dir`` and writes it (see the module docstring).
    Never raises for "can't do it here": returns ``{"status": "skipped", "reason": ...}`` instead, so a missing
    tool costs the dense step only, not the camera solve before it."""
    o = options(raw_options) or options(True)
    progress = progress or (lambda f, m="": None)
    log = log or (lambda m: None)
    dataset_dir = Path(dataset_dir)
    t0 = time.perf_counter()
    cams = load_cameras(dataset_dir, image_root, mask_dirs)
    sparse_path, sparse_xyz, sparse_rgb = _sparse(dataset_dir)
    if len(cams) < 2:
        return _done(dataset_dir, {"status": "skipped", "reason": "fewer than 2 training photos with their images"}, t0)
    avail = available_methods()
    has_sensor = sum(c.depth is not None for c in cams) >= 0.5 * len(cams)
    order = {"auto": ["mvs", "depth"], "mvs": ["mvs", "depth"], "depth": ["depth"]}[o["method"]]
    notes: List[str] = []
    method = None
    for m in order:
        if m == "mvs":
            if not avail["mvs"]:
                notes.append("multi-view stereo needs COLMAP with CUDA (an NVIDIA GPU)")
                continue
            if len(sparse_xyz) < MVS_MIN_POINTS:
                notes.append(f"multi-view stereo needs at least {MVS_MIN_POINTS} sparse points (the dataset has {len(sparse_xyz)})")
                continue
        if m == "depth" and not has_sensor:
            if not avail["predictedDepth"]:
                notes.append("no depth maps in the dataset, and predicting depth needs PyTorch + transformers (the Models pack)")
                continue
            if len(sparse_xyz) < 100:
                notes.append("predicted depth needs at least 100 sparse points to be scaled")
                continue
        method = m
        break
    if method is None:
        return _done(dataset_dir, {"status": "skipped", "reason": "; ".join(notes) or "no method available"}, t0)
    if o["method"] == "mvs" and method != "mvs":
        log(f"dense points: {notes[0]} — fusing depth maps instead")
    work = Path(work_dir) if work_dir else dataset_dir / ".dense_work"
    work.mkdir(parents=True, exist_ok=True)
    depths = median_depths(cams, sparse_xyz, fallback=1.0)
    info: dict = {"status": "ok", "method": method, "numCameras": len(cams), "sparsePoints": int(len(sparse_xyz))}
    try:
        if method == "mvs":
            used = write_mvs_model(cams, sparse_xyz, sparse_rgb, work / "model")
            if len(used) < 2:
                raise SplatPrepError("too few photos see the sparse points for multi-view stereo")
            fused = run_colmap_mvs(avail["mvsTool"], colmap_cli_cuda(), link_mvs_images(cams, used, work / "images"), work / "model", work, o, log_path,
                                   progress, len(used))
            xyz, rgb = read_points_ply(fused)
            info["rawPoints"] = int(len(xyz))
            xyz, rgb, n_sky = drop_sky_points(xyz, rgb, cams)
            voxel = o["voxelSize"] or _auto_voxel([cams[i] for i in used], depths[used], o["maxImageSize"]) / 2
            xyz, rgb, voxel = voxel_downsample(xyz, rgb, voxel, o["maxPoints"])
            info.update(tool=avail["mvsTool"], skyPointsDropped=n_sky)
        else:
            maps, source = build_depth_maps(cams, sparse_xyz, o, device, progress, log)
            if len(maps) < 2:
                return _done(dataset_dir, {"status": "skipped", "reason": f"only {len(maps)} usable {source} depth map(s)"}, t0)
            long_side = max(max(d.shape) for d in maps.values())
            nb = choose_neighbours(cams, depths, o["neighbours"])
            voxel = o["voxelSize"] or _auto_voxel(cams, depths, long_side)
            tol = 0.03 if source == "sensor" else 0.06
            xyz, rgb, voxel, raw = fuse_depth_maps(cams, maps, nb, o, tol, voxel, progress)
            info.update(depthSource=source, depthMaps=len(maps), rawPoints=raw, tolerance=tol)
    finally:
        if work_dir is None:
            shutil.rmtree(work, ignore_errors=True)
    if len(xyz) == 0:
        return _done(dataset_dir, {**info, "status": "skipped", "reason": "no points agreed between photos"}, t0)
    out_xyz, out_rgb = xyz, rgb
    if o["mode"] == "add" and len(sparse_xyz):
        out_xyz = np.concatenate([sparse_xyz, xyz])
        out_rgb = np.concatenate([sparse_rgb, rgb])
        if len(out_xyz) > o["maxPoints"]:
            out_xyz, out_rgb, voxel = voxel_downsample(out_xyz, out_rgb, voxel, o["maxPoints"])
    pts = dataset_dir / "points3d.ply"
    if sparse_path is not None and sparse_path.resolve() == pts.resolve():
        shutil.copy2(pts, dataset_dir / "points3d_sparse.ply")
    write_points_ply(pts, out_xyz, out_rgb)
    ds = json.loads((dataset_dir / "dataset.json").read_text(encoding="utf-8"))
    ds["points"] = "points3d.ply"
    (dataset_dir / "dataset.json").write_text(json.dumps(ds, indent=2))
    info.update(numPoints=int(len(out_xyz)), densePoints=int(len(xyz)), voxelSize=float(voxel), mode=o["mode"])
    if notes and method != order[0]:
        info["note"] = notes[0]
    progress(1.0, f"dense points: {len(out_xyz):,} ({method})")
    return _done(dataset_dir, info, t0)


def mvs_name(i: int, cam: Cam) -> str:
    return f"{i:06d}{cam.image.suffix.lower()}"


def link_mvs_images(cams: Sequence[Cam], used: Sequence[int], out: Path) -> Path:
    """One folder with the MVS model's photos under their model names (links; copies where links can't be made)."""
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    for i in used:
        dst = out / mvs_name(i, cams[i])
        try:
            os.symlink(cams[i].image.resolve(), dst)
        except OSError:
            shutil.copy2(cams[i].image, dst)
    return out


def _done(dataset_dir: Path, info: dict, t0: float) -> dict:
    info["seconds"] = round(time.perf_counter() - t0, 1)
    try:
        (Path(dataset_dir) / "dense.json").write_text(json.dumps(info, indent=1))
    except OSError:
        pass
    return info
