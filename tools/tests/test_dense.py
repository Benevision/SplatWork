"""Dense points (splatprep/dense.py): depth-map fusion, COLMAP MVS orchestration (fake colmap), request wiring."""
import json
import os
import stat
import sys
from pathlib import Path

import numpy as np
import pytest

from splatprep import dense
from splatprep.colmap_model import read_binary_model
from splatprep.errors import RequestError
from splatprep.ply import read_points_ply, write_points_ply
from splatprep.runner import normalize_request, run_request


@pytest.fixture(scope="module")
def synth_depth(tmp_path_factory):
    """Orbit with sensor-like depth maps (float16 Z, as an iPhone capture carries)."""
    from splatprep.synth import generate

    out = tmp_path_factory.mktemp("dense") / "orbit"
    generate(out, "orbit", num_views=10, width=160, height=120, seed=5, supersample=1, photometric_strength=0.2,
             write_depth=True)
    return out


def _copy(src: Path, dst: Path) -> Path:
    import shutil

    shutil.copytree(src, dst)
    return dst


def _gt_depths(ds: Path):
    doc = json.loads((ds / "dataset.json").read_text())
    out = []
    for c in doc["cameras"]:
        d = np.fromfile(ds / c["depth"], dtype="<f2").reshape(c["depthHeight"], c["depthWidth"]).astype(np.float64)
        K = np.array([[c["fx"], 0, c["cx"]], [0, c["fy"], c["cy"]], [0, 0, 1]])
        out.append((d, K, np.asarray(c["cameraToWorld"], float)))
    return out


def _on_surface(xyz: np.ndarray, gts, tol=0.03) -> np.ndarray:
    """Per point: some camera sees it in front of its GT depth within ``tol`` (relative)."""
    best = np.full(len(xyz), np.inf)
    for d, K, c2w in gts:
        w2c = np.linalg.inv(c2w)
        Xc = xyz @ w2c[:3, :3].T + w2c[:3, 3]
        z = Xc[:, 2]
        zs = np.where(z > 1e-6, z, 1)
        u = (K[0, 0] * Xc[:, 0] / zs + K[0, 2]).astype(int)
        v = (K[1, 1] * Xc[:, 1] / zs + K[1, 2]).astype(int)
        h, w = d.shape
        ok = (z > 1e-6) & (u >= 0) & (u < w) & (v >= 0) & (v < h)
        rel = np.full(len(xyz), np.inf)
        g = d[v[ok], u[ok]]
        rel[np.nonzero(ok)[0]] = np.where(g > 0, np.abs(z[ok] - g) / np.maximum(g, 1e-9), np.inf)
        best = np.minimum(best, rel)
    return best <= tol


def test_options_validation():
    assert dense.options(None) is None and dense.options(False) is None
    assert dense.options({"enabled": False}) is None
    o = dense.options(True)
    assert o["method"] == "auto" and o["mode"] == "replace" and o["maxPoints"] == 1_000_000
    o = dense.options({"maxPoints": 5, "minViews": 1, "neighbours": 0})
    assert o["maxPoints"] == 10_000 and o["minViews"] == 2 and o["neighbours"] == 1
    with pytest.raises(Exception):
        dense.options({"method": "magic"})
    with pytest.raises(RequestError):
        normalize_request({"task": "sfm", "dense": {"mode": "both"}})
    assert normalize_request({"task": "dense"})["task"] == "dense"


def test_voxel_grid_caps_and_averages():
    rng = np.random.default_rng(0)
    xyz = rng.random((200_000, 3))
    rgb = np.full((len(xyz), 3), 100, np.uint8)
    p, c, v = dense.voxel_downsample(xyz, rgb, 0.001, 20_000)
    assert len(p) <= 20_000 and v > 0.001
    assert np.all(c == 100)
    # two points in one cell become their mean
    g = dense.VoxelGrid(1.0, 10)
    g.add(np.array([[0.2, 0.2, 0.2], [0.4, 0.4, 0.4]]), np.array([[0, 0, 0], [200, 200, 200]]))
    p, c = g.points()
    assert len(p) == 1 and np.allclose(p[0], 0.3) and c[0, 0] == 100
    # the grid never holds more than its limit while adding
    g = dense.VoxelGrid(1e-4, 1000)
    g.add(rng.random((50_000, 3)), np.zeros((50_000, 3)))
    assert len(g) <= 1000


def test_neighbours_prefer_nearby_views(synth_depth):
    cams = dense.load_cameras(synth_depth)
    nb = dense.choose_neighbours(cams, np.full(len(cams), 4.0), 2)
    # an orbit: each camera's best neighbours are the ones next to it on the circle
    for i, n in enumerate(nb):
        ring = {(i - 1) % len(cams), (i + 1) % len(cams), (i - 2) % len(cams), (i + 2) % len(cams)}
        assert n and set(n) <= ring


def test_sensor_depth_fusion(synth_depth, tmp_path):
    ds = _copy(synth_depth, tmp_path / "ds")
    sparse_n = len(read_points_ply(ds / "points3d.ply")[0])
    info = dense.densify(ds, {"maxPoints": 30_000})
    assert info["status"] == "ok", info
    assert info["method"] == "depth" and info["depthSource"] == "sensor"
    xyz, rgb = read_points_ply(ds / "points3d.ply")
    assert 1000 < len(xyz) <= 30_000
    assert info["numPoints"] == len(xyz)
    assert len(read_points_ply(ds / "points3d_sparse.ply")[0]) == sparse_n
    assert _on_surface(xyz, _gt_depths(ds)).mean() > 0.95
    assert json.loads((ds / "dense.json").read_text())["status"] == "ok"
    assert not (ds / ".dense_work").exists()


def test_inconsistent_depth_is_rejected(synth_depth, tmp_path):
    """One photo's depth map is wrong (twice as far): its points don't agree with the neighbours and are dropped."""
    ds = _copy(synth_depth, tmp_path / "ds")
    doc = json.loads((ds / "dataset.json").read_text())
    bad = doc["cameras"][3]
    d = np.fromfile(ds / bad["depth"], dtype="<f2")
    (d.astype(np.float32) * 2).astype("<f2").tofile(ds / bad["depth"])
    info = dense.densify(ds, {"maxPoints": 30_000})
    assert info["status"] == "ok"
    xyz, _ = read_points_ply(ds / "points3d.ply")
    assert _on_surface(xyz, [g for k, g in enumerate(_gt_depths(synth_depth))]).mean() > 0.95


def test_masks_and_holdout_give_no_points(synth_depth, tmp_path):
    from PIL import Image

    ds = _copy(synth_depth, tmp_path / "ds")
    doc = json.loads((ds / "dataset.json").read_text())
    (ds / "masks").mkdir()
    for c in doc["cameras"]:
        m = np.full((c["height"], c["width"]), 255, np.uint8)
        m[:, : c["width"] // 2] = 0          # left half masked out in every photo
        Image.fromarray(m).save(ds / "masks" / (Path(c["image"]).stem + ".png"))
        c["mask"] = f"masks/{Path(c['image']).stem}.png"
    doc["cameras"][0]["holdout"] = True
    (ds / "dataset.json").write_text(json.dumps(doc))
    assert len(dense.load_cameras(ds)) == len(doc["cameras"]) - 1
    full = dense.densify(_copy(synth_depth, tmp_path / "full"), {"maxPoints": 30_000})
    half = dense.densify(ds, {"maxPoints": 30_000})
    assert half["status"] == "ok" and half["rawPoints"] < 0.75 * full["rawPoints"]


def test_add_mode_keeps_sparse_points(synth_depth, tmp_path):
    ds = _copy(synth_depth, tmp_path / "ds")
    sparse = read_points_ply(ds / "points3d.ply")[0]
    info = dense.densify(ds, {"mode": "add", "maxPoints": 1_000_000})
    assert info["status"] == "ok" and info["numPoints"] == len(sparse) + info["densePoints"]


def test_skipped_without_depth_or_tools(synth_small, tmp_path, monkeypatch):
    """No depth maps, no depth model, no CUDA COLMAP: the dense step says why and leaves the points alone."""
    ds = _copy(synth_small, tmp_path / "ds")
    before = (ds / "points3d.ply").read_bytes()
    monkeypatch.setattr(dense, "available_methods", lambda: {"mvs": False, "mvsTool": None, "depth": True, "predictedDepth": False})
    info = dense.densify(ds, True)
    assert info["status"] == "skipped" and "CUDA" in info["reason"] and "depth" in info["reason"]
    assert (ds / "points3d.ply").read_bytes() == before
    # through a request (task = dense): reported, not an error
    stats = run_request({"task": "dense", "datasetDir": str(ds)}, tmp_path)
    assert stats["task"] == "dense" and stats["status"] == "skipped"


def test_dense_task_needs_a_dataset(tmp_path):
    with pytest.raises(RequestError):
        run_request({"task": "dense", "datasetDir": "nothing"}, tmp_path)


def test_predicted_depth_fusion(synth_small, tmp_path, monkeypatch):
    """Predicted depth (stand-in model: the true inverse depth, scaled and offset) is fitted to the sparse points."""
    ds = _copy(synth_small, tmp_path / "ds")
    from splatprep.synth import render

    doc = json.loads((ds / "dataset.json").read_text())
    truth = {}
    for c in doc["cameras"]:
        truth[Path(c["image"]).name] = c

    class FakePredictor:
        def __init__(self, device):
            pass

        def __call__(self, rgb, size):
            # rgb is the resized photo; find its camera by matching the call order
            c = order.pop(0)
            K = np.array([[c["fx"], 0, c["cx"]], [0, c["fy"], c["cy"]], [0, 0, 1.0]])
            from splatprep.synth import make_textures

            _, d = render(np.asarray(c["cameraToWorld"]), K, c["width"], c["height"], textures, 1)
            from PIL import Image

            inv = np.where(np.isfinite(d), 1.0 / d, 0.0).astype(np.float32)
            inv = np.asarray(Image.fromarray(inv).resize(size, Image.NEAREST))
            return 3.0 * inv + 0.01

        def close(self):
            pass

    from splatprep.synth import make_textures

    textures = make_textures(np.random.default_rng(3 + 1000))
    order = [c for c in doc["cameras"] if not c.get("holdout")]
    monkeypatch.setattr(dense, "DepthPredictor", FakePredictor)
    monkeypatch.setattr(dense, "depth_model_available", lambda: True)
    monkeypatch.setattr(dense, "available_methods", lambda: {"mvs": False, "mvsTool": None, "depth": True, "predictedDepth": True})
    info = dense.densify(ds, {"maxPoints": 20_000})
    assert info["status"] == "ok", info
    assert info["depthSource"] == "predicted" and info["depthMaps"] >= 6
    xyz, _ = read_points_ply(ds / "points3d.ply")
    gts = []
    for c in doc["cameras"]:
        K = np.array([[c["fx"], 0, c["cx"]], [0, c["fy"], c["cy"]], [0, 0, 1.0]])
        _, d = render(np.asarray(c["cameraToWorld"]), K, c["width"], c["height"], textures, 1)
        gts.append((np.where(np.isfinite(d), d, 0), K, np.asarray(c["cameraToWorld"], float)))
    assert _on_surface(xyz, gts, tol=0.05).mean() > 0.9


def test_fit_inverse_depth_recovers_scale():
    rng = np.random.default_rng(1)
    z = rng.uniform(1, 10, 500)
    pred = (1 / z - 0.02) / 0.5          # 1/z = 0.5·pred + 0.02
    pred[:40] = rng.uniform(0, 3, 40)    # outliers (occluded points)
    a, b, err = dense.fit_inverse_depth(pred, z)
    assert abs(a - 0.5) < 0.02 and abs(b - 0.02) < 0.01 and err < 0.01


FAKE_COLMAP = r'''#!{python}
import json, os, struct, sys
from pathlib import Path
args = sys.argv[1:]
with open(os.environ["FAKE_COLMAP_LOG"], "a") as f:
    f.write(json.dumps(args) + "\n")
def opt(n):
    return args[args.index("--" + n) + 1]
if not args or args[0] in ("-h", "help"):
    print("COLMAP 4.0 (Commit fake with CUDA)\nUsage:\n  colmap [command] [options]\n  image_undistorter\n  patch_match_stereo\n  stereo_fusion")
    sys.exit(0)
if "-h" in args:
    opts = {"patch_match_stereo": "--workspace_path --PatchMatchStereo.max_image_size --PatchMatchStereo.cache_size --PatchMatchStereo.geom_consistency",
            "stereo_fusion": "--workspace_path --output_path --StereoFusion.cache_size --StereoFusion.num_threads --StereoFusion.max_image_size"}
    print(opts.get(args[0], ""))
    sys.exit(0)
if args[0] == "image_undistorter":
    Path(opt("output_path")).mkdir(parents=True)
elif args[0] == "patch_match_stereo":
    print("Processing view 1 / 2 for a"); print("Processing view 2 / 2 for b")
elif args[0] == "stereo_fusion":
    pts = [(x * 0.01, 0.0, 1.0) for x in range(300)]
    hdr = ("ply\nformat binary_little_endian 1.0\nelement vertex %d\nproperty float x\nproperty float y\nproperty float z\n"
           "property float nx\nproperty float ny\nproperty float nz\nproperty uchar red\nproperty uchar green\n"
           "property uchar blue\nend_header\n") % len(pts)
    with open(opt("output_path"), "wb") as f:
        f.write(hdr.encode())
        for p in pts:
            f.write(struct.pack("<6f3B", *p, 0, 0, 1, 10, 20, 30))
'''


def test_mvs_orchestration_with_fake_colmap(synth_small, tmp_path, monkeypatch):
    exe = tmp_path / "colmap"
    exe.write_text(FAKE_COLMAP.replace("{python}", sys.executable))
    exe.chmod(exe.stat().st_mode | stat.S_IEXEC)
    log = tmp_path / "calls.jsonl"
    monkeypatch.setenv("SPLATPREP_COLMAP", str(exe))
    monkeypatch.setenv("FAKE_COLMAP_LOG", str(log))
    dense._CUDA_CACHE.clear()
    from splatprep import mapper

    mapper.colmap_options.cache_clear()
    assert dense.available_methods()["mvsTool"] == "colmap"
    ds = _copy(synth_small, tmp_path / "ds")
    work = tmp_path / "work"
    info = dense.densify(ds, {"maxImageSize": 800}, work_dir=work)
    assert info["status"] == "ok", info
    assert info["method"] == "mvs" and info["rawPoints"] == 300
    calls = [json.loads(l) for l in log.read_text().splitlines() if "-h" not in l and l.strip() not in ('["help"]',)]
    cmds = [c[0] for c in calls if c and c[0] not in ("-h", "help")]
    assert cmds == ["image_undistorter", "patch_match_stereo", "stereo_fusion"]
    pm = calls[[c[0] for c in calls].index("patch_match_stereo")]
    assert pm[pm.index("--PatchMatchStereo.max_image_size") + 1] == "800"
    assert int(pm[pm.index("--PatchMatchStereo.cache_size") + 1]) <= 32
    fu = calls[[c[0] for c in calls].index("stereo_fusion")]
    assert "--StereoFusion.num_threads" in fu
    # the workspace model: every photo, PINHOLE, with tracks from the sparse points
    rec = read_binary_model(work / "model")
    n_train = sum(1 for c in json.loads((ds / "dataset.json").read_text())["cameras"] if not c.get("holdout"))
    assert len(rec.images) == n_train and all(c.model == "PINHOLE" for c in rec.cameras.values())
    assert len(rec.points3D) > 100 and all(len(p.image_ids) >= 2 for p in rec.points3D.values())
    assert all((work / "images" / im.name).exists() for im in rec.images.values())
    xyz, rgb = read_points_ply(ds / "points3d.ply")
    assert 0 < len(xyz) <= 300 and tuple(rgb[0]) == (10, 20, 30)
    dense._CUDA_CACHE.clear()


def test_sfm_request_runs_dense_after_export(synth_depth, tmp_path, monkeypatch):
    """run_sfm → export → dense: the exported dataset's points3d.ply is the dense cloud, stats say so."""
    from splatprep import runner

    calls = {}

    def fake_densify(out_dir, opts, progress, log, **kw):
        calls.update(out_dir=Path(out_dir), opts=opts, kw=kw)
        progress(0.5, "half")
        return {"status": "ok", "method": "depth", "numPoints": 7, "seconds": 0.1}

    monkeypatch.setattr(dense, "densify", fake_densify)
    from splatprep.progress import StagedProgress

    events = []
    sp = StagedProgress(events.append, [("export", 1.0), ("dense", 0.25)])
    (tmp_path / "masks").mkdir()
    info = runner.run_dense_step(dense.options(True), synth_depth, tmp_path / "images",
                                 {"exportImages": False, "masksDir": "masks"}, tmp_path, "cpu", sp)
    assert info["numPoints"] == 7 and calls["out_dir"] == synth_depth
    assert calls["kw"]["image_root"] == tmp_path / "images"
    assert calls["kw"]["mask_dirs"] == [("mask", tmp_path / "masks")]
    assert any("dense points: 7" in e["message"] for e in events)

    def broken(*a, **k):
        raise MemoryError("out of memory")

    monkeypatch.setattr(dense, "densify", broken)
    ds = _copy(synth_depth, tmp_path / "ds")
    info = runner.run_dense_step(dense.options(True), ds, ds, {}, tmp_path, "cpu", sp)
    assert info["status"] == "failed" and "memory" in info["reason"]
    assert json.loads((ds / "dense.json").read_text())["status"] == "failed"
