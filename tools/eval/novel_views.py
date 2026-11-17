#!/usr/bin/env python3
"""Off-angle quality: a repeatable holdout-sector score (ROADMAP issue Q11), and a by-eye sheet of novel views.

**Sector score** (the measurement; `sector` and `table` sub-commands). A contiguous sector of cameras (by the
angle of the camera around the scene, or of its view direction for inside-out captures such as rooms) is held
out of training together with the dataset's ordinary every-Nth held-out photos. The model is trained with the
Mac headless CLI and scored on

* the sector (PSNR / SSIM): views from an angle no training photo was taken from, with real photos as truth;
* the ordinary held-out photos outside the sector ("normal" PSNR / SSIM): must not get worse;
* geometry indicators that need no ground truth, read from the splat file: share of splats whose longest axis
  is more than 10 x the shortest, share of needles (longest more than 5 x the middle axis), mean opacity, share
  of splats under 0.05 opacity, and thickness along the splat's normal (shortest axis, absolute and relative to
  the other two).

    python3 tools/eval/novel_views.py sector --dataset <dataset> --app mac/.build/release/SplatStudio \
        --out q11/van --name baseline [--config overrides.json] [--preset balanced] [--width 40]
    python3 tools/eval/novel_views.py table q11/van            # markdown table of every run in the folder

**Sheet** (no sub-command; the older by-eye comparison from viewpoints no photo was taken from).

Takes held-out cameras of a dataset, moves each forward along its view direction by ``--closer`` × the median
depth of the scene points it sees (so a drone photo becomes a lower, zoomed-in look at the ground), writes those
cameras as a render-only dataset, renders every model from them with ``SplatStudio --render`` and lays the
results side by side (no ground truth exists for these views; judge floaters, haze and sharpness by eye).

    python3 tools/eval/novel_views.py --dataset <dataset> --model ours=a.ply --model spirula=b.ply \
        --app mac/.build/release/SplatStudio --out novel_out [--views 6] [--closer 0.6]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def read_ply(path: Path) -> np.ndarray:
    """Vertex table of a binary little-endian PLY as a structured array."""
    f = open(path, "rb")
    props, n, fmt = [], 0, ""
    while True:
        s = f.readline().decode().strip()
        if s.startswith("format"):
            fmt = s.split()[1]
        if s.startswith("element vertex"):
            n = int(s.split()[2])
        if s.startswith("property"):
            props.append((s.split()[1], s.split()[2]))
        if s == "end_header":
            break
    if fmt != "binary_little_endian":
        raise SystemExit("points PLY must be binary little-endian")
    tmap = {"float": "<f4", "double": "<f8", "uchar": "u1", "uint8": "u1", "int": "<i4", "uint": "<u4", "float32": "<f4"}
    dt = np.dtype([(name, tmap[t]) for t, name in props])
    return np.frombuffer(f.read(n * dt.itemsize), dt)


def read_points(path: Path) -> np.ndarray:
    a = read_ply(path)
    return np.stack([a["x"], a["y"], a["z"]], 1).astype(np.float64)


# ---------------------------------------------------------------------------------------------- sector score

def geometry_indicators(ply: Path) -> dict:
    """Ground-truth-free shape statistics of a 3DGS splat file (log scales, logit opacity)."""
    a = read_ply(ply)
    s = np.sort(np.exp(np.stack([a["scale_0"], a["scale_1"], a["scale_2"]], 1).astype(np.float64)), axis=1)
    o = 1.0 / (1.0 + np.exp(-a["opacity"].astype(np.float64)))
    lo, mid, hi = s[:, 0], s[:, 1], s[:, 2]
    return {
        "splats": int(len(a)),
        "ratio10": float(np.mean(hi / lo > 10)),             # longest axis > 10 x shortest (needles and thin discs)
        "needles": float(np.mean(hi / mid > 5)),             # longest axis > 5 x the middle one (true needles)
        "opacity_mean": float(o.mean()),
        "opacity_lt_005": float(np.mean(o < 0.05)),
        "opacity_gt_05": float(np.mean(o > 0.5)),
        "thickness_mm": float(np.median(lo) * 1000),         # along the normal; metres assumed (LiDAR captures)
        "thickness_rel": float(np.median(lo / np.sqrt(mid * hi))),   # 1 = ball, towards 0 = flat disc
    }


def pick_sector(ds: dict, width: float, fraction: float, by: str) -> tuple[set, str]:
    """Ids of the cameras inside one `width`-degree sector: the sector (5-degree steps round the scene) whose
    camera count is nearest to `fraction` of all cameras, so every scene loses a similar share of its photos."""
    cams = ds["cameras"]
    up = np.array(ds.get("upVector") or ds.get("gravityUp") or [0, -1, 0], dtype=np.float64)
    up /= np.linalg.norm(up)
    M = np.array([c["cameraToWorld"] for c in cams], dtype=np.float64)
    eye, fwd = M[:, :3, 3], M[:, :3, 2]
    centre = np.array(ds.get("sceneCenter") or eye.mean(0), dtype=np.float64)
    if by in ("low", "high"):   # a height band instead of a sector: views from below (or above) every training photo
        h = eye @ up if by == "high" else -(eye @ up)
        inside = h >= np.sort(h)[-max(1, round(fraction * len(cams)))]
        gap = np.sort(h)[-int(inside.sum())] - np.median(h)
        return ({cams[i]["id"] for i in np.flatnonzero(inside)},
                f"{int(inside.sum())} of {len(cams)} cameras, the {by}est by height ({gap:.2f} scene units from the median)")
    if by == "auto":      # an orbit looks inwards; a room capture looks outwards
        inward = np.mean(np.einsum("ij,ij->i", fwd, centre - eye) > 0)
        by = "position" if inward > 0.7 else "direction"
    v = (eye - centre) if by == "position" else fwd
    v = v - np.outer(v @ up, up)
    ref = np.cross(up, [1.0, 0, 0]) if abs(up[0]) < 0.9 else np.cross(up, [0, 1.0, 0])
    ref /= np.linalg.norm(ref)
    ang = np.degrees(np.arctan2(v @ np.cross(up, ref), v @ ref))
    best = None
    for c0 in range(-180, 180, 5):
        inside = np.abs((ang - c0 + 180) % 360 - 180) <= width / 2
        miss = abs(int(inside.sum()) - fraction * len(cams))
        if best is None or miss < best[0]:
            best = (miss, c0, inside)
    _, c0, inside = best
    return ({cams[i]["id"] for i in np.flatnonzero(inside)},
            f"{int(inside.sum())} of {len(cams)} cameras, {width:.0f} deg sector by {by} centred {c0} deg")


def write_split(src: Path, out: Path, ds: dict, sector: set) -> dict:
    """train / sector / normal copies of the dataset: same files (symlinks), different held-out flags."""
    dirs = {}
    for name in ("train", "sector", "normal", "census"):
        d = out / "split" / name
        d.mkdir(parents=True, exist_ok=True)
        for item in src.iterdir():
            link = d / item.name
            if item.name != "dataset.json" and not link.exists():
                os.symlink(item.resolve(), link)
        cams = []
        for c in ds["cameras"]:
            c2 = dict(c)
            ins, was = c["id"] in sector, bool(c.get("holdout"))
            if name == "train":
                c2["holdout"] = ins or was
            elif name == "census":        # training photos only: pruning must not see the held-out views
                if ins or was:
                    continue
            elif name == "sector":
                if not ins:
                    continue
                c2["holdout"] = True
            else:
                if ins or not was:
                    continue
                c2["holdout"] = True
            cams.append(c2)
        meta = dict(ds); meta["cameras"] = cams
        (d / "dataset.json").write_text(json.dumps(meta))
        dirs[name] = d
    return dirs


def score_views(app: Path, ply: Path, dataset: Path, renders: Path, max_size: int) -> dict:
    """Held-out PSNR from the CLI's own evaluation, SSIM from its saved renders."""
    sys.path.insert(0, str(Path(__file__).parent))
    from evaluate import load_rgb, ssim
    r = subprocess.run([str(app), "--eval", str(ply), "--dataset", str(dataset), "--save-renders", str(renders),
                        "--max-size", str(max_size)], capture_output=True, text=True)
    m = re.search(r"held-out PSNR ([\d.]+) dB over (\d+) views", r.stdout)
    if not m:
        raise SystemExit(f"evaluation failed:\n{r.stdout}\n{r.stderr}")
    ds = json.loads((dataset / "dataset.json").read_text())
    vals = []
    for c in ds["cameras"]:
        f = renders / Path(c["image"]).name
        if f.is_file():
            im = load_rgb(f)
            vals.append(ssim(im, load_rgb(dataset / c["image"], (im.shape[1], im.shape[0]))))
    return {"psnr": float(m.group(1)), "views": int(m.group(2)), "ssim": float(np.mean(vals)) if vals else float("nan")}


COLUMNS = [("sector PSNR", "sector.psnr", "{:.2f}"), ("sector SSIM", "sector.ssim", "{:.4f}"),
           ("normal PSNR", "normal.psnr", "{:.2f}"), ("normal SSIM", "normal.ssim", "{:.4f}"),
           ("splats", "geometry.splats", "{:,}"), ("ratio > 10", "geometry.ratio10", "{:.1%}"),
           ("needles", "geometry.needles", "{:.1%}"), ("mean opacity", "geometry.opacity_mean", "{:.3f}"),
           ("opacity < 0.05", "geometry.opacity_lt_005", "{:.1%}"), ("thickness mm", "geometry.thickness_mm", "{:.2f}"),
           ("thickness rel", "geometry.thickness_rel", "{:.3f}"), ("train min", "train_minutes", "{:.1f}")]


def table(folder: Path) -> str:
    runs = sorted((json.loads(p.read_text()) for p in folder.glob("*.json")), key=lambda r: r.get("finished", 0))
    lines = ["| run | " + " | ".join(c[0] for c in COLUMNS) + " |", "|---|" + "---:|" * len(COLUMNS)]
    for r in runs:
        cells = []
        for _, key, fmt in COLUMNS:
            v = r
            for k in key.split("."):
                v = v.get(k) if isinstance(v, dict) else None
            cells.append(fmt.format(v) if v is not None else "–")
        lines.append(f"| {r['name']} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def sector_main(argv):
    ap = argparse.ArgumentParser(prog="novel_views.py sector")
    ap.add_argument("--dataset", required=True, type=Path)
    ap.add_argument("--app", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--name", default="baseline")
    ap.add_argument("--config", type=Path, help="TrainingConfig overrides (JSON) on top of the preset")
    ap.add_argument("--preset", default="balanced")
    ap.add_argument("--strategy", default="adaptive", help="adaptive (default: MCMC collapses above ~1 M, issue Q1)")
    ap.add_argument("--iterations", type=int)
    ap.add_argument("--width", type=float, default=40, help="angular width of the held-out sector, degrees")
    ap.add_argument("--fraction", type=float, default=0.10, help="the sector holding nearest this share of the cameras is used")
    ap.add_argument("--view-clean", type=float, metavar="F", help="with --from: prune run FROM's splat by contribution "
                    "across the training photos (invisible, or seen by under F x the typical number of photos) and score that")
    ap.add_argument("--from", dest="source", help="name of an earlier run in --out")
    ap.add_argument("--by", default="auto", choices=["auto", "position", "direction", "low", "high"],
                    help="low / high: hold out the lowest / highest --fraction of the cameras instead of a sector")
    ap.add_argument("--thin", type=int, default=1, help="hold out only every Nth camera of the sector / height band "
                    "(the rest train): how much of the gap is missing coverage")
    ap.add_argument("--max-size", type=int, default=1600, help="longest side of the scored renders")
    a = ap.parse_args(argv)
    ds = json.loads((a.dataset / "dataset.json").read_text())
    sector, how = pick_sector(ds, a.width, a.fraction, a.by)
    if a.thin > 1:   # coverage test: only every Nth picked camera stays held out, the rest train
        keep = set(sorted(sector)[::a.thin])
        how += f"; 1 in {a.thin} held out ({len(keep)}), the other {len(sector) - len(keep)} train"
        sector = keep
    dirs = write_split(a.dataset, a.out, ds, sector)
    cfg = {"strategy": a.strategy}
    if a.config:
        cfg.update(json.loads(a.config.read_text()))
    cfg_file = a.out / f"{a.name}.config"
    cfg_file.write_text(json.dumps(cfg))
    ply = a.out / f"{a.name}.ply"
    minutes = None
    if a.view_clean is not None and a.source and not ply.is_file():
        r = subprocess.run([str(a.app), "--eval", str(a.out / f"{a.source}.ply"), "--dataset", str(dirs["census"]),
                            "--view-clean", str(a.view_clean), "--clean-out", str(ply)], capture_output=True, text=True)
        print("\n".join(l for l in r.stdout.splitlines() if "clean" in l))
        cfg = {"from": a.source, "viewClean": a.view_clean}
    if not ply.is_file():
        cmd = [str(a.app), "--train", str(dirs["train"]), "--preset", a.preset, "--config", str(cfg_file), "--out", str(ply)]
        if a.iterations:
            cmd += ["--iterations", str(a.iterations)]
        t0 = time.time()
        with open(a.out / f"{a.name}.log", "w") as log:
            if subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT).returncode != 0 or not ply.is_file():
                raise SystemExit(f"training failed, see {a.out / (a.name + '.log')}")
        minutes = (time.time() - t0) / 60
    res = {"name": a.name, "dataset": a.dataset.resolve().parent.name, "sector_cameras": how, "preset": a.preset,
           "iterations": a.iterations, "config": cfg, "train_minutes": minutes, "geometry": geometry_indicators(ply)}
    for part in ("sector", "normal"):
        res[part] = score_views(a.app, ply, dirs[part], a.out / "renders" / a.name / part, a.max_size)
    res["finished"] = time.time()
    (a.out / f"{a.name}.json").write_text(json.dumps(res, indent=1))
    print(f"sector: {how}")
    print(table(a.out))


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "sector":
        return sector_main(sys.argv[2:])
    if len(sys.argv) > 2 and sys.argv[1] == "table":
        return print(table(Path(sys.argv[2])))
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, type=Path)
    ap.add_argument("--model", action="append", required=True, help="name=splat.ply")
    ap.add_argument("--app", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--views", type=int, default=6)
    ap.add_argument("--closer", type=float, default=0.6)
    ap.add_argument("--width", type=int, default=520)
    a = ap.parse_args()
    ds = json.loads((a.dataset / "dataset.json").read_text())
    pts = read_points(a.dataset / ds["points"]) if ds.get("points") else None
    held = [c for c in ds["cameras"] if c.get("holdout")] or ds["cameras"]
    picks = [held[int(i)] for i in np.linspace(0, len(held) - 1, min(a.views, len(held)))]
    cams = []
    for k, c in enumerate(picks):
        M = np.array(c["cameraToWorld"], dtype=np.float64)
        eye, fwd = M[:3, 3], M[:3, 2] / np.linalg.norm(M[:3, 2])
        depth = None
        if pts is not None:
            rel = pts - eye
            z = rel @ fwd
            lateral = np.linalg.norm(rel - np.outer(z, fwd), axis=1)
            sel = (z > 0) & (lateral < 0.25 * z)          # points near the middle of the view
            if sel.sum() > 20:
                depth = float(np.median(z[sel]))
        if depth is None:
            depth = float(np.linalg.norm(np.array(ds.get("sceneCenter", [0, 0, 0])) - eye))
        M2 = M.copy()
        M2[:3, 3] = eye + fwd * depth * a.closer
        c2 = dict(c); c2["id"] = k; c2["cameraToWorld"] = M2.tolist(); c2.pop("holdout", None)
        cams.append(c2)
    a.out.mkdir(parents=True, exist_ok=True)
    nd = a.out / "dataset"
    nd.mkdir(exist_ok=True)
    meta = {k: v for k, v in ds.items() if k not in ("cameras", "points")}
    meta["cameras"] = cams
    (nd / "dataset.json").write_text(json.dumps(meta))
    rows = []
    for k, c in enumerate(cams):
        row = []
        for m in a.model:
            name, _, ply = m.partition("=")
            png = a.out / f"{name}_{k}.png"
            subprocess.run([str(a.app), "--render", ply, str(png), "--dataset", str(nd), "--view", str(k)],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
            if png.is_file():
                im = Image.open(png).convert("RGB")
                im = im.resize((a.width, int(a.width * im.height / im.width)), Image.LANCZOS)
                d = ImageDraw.Draw(im)
                d.rectangle([0, 0, 8 + 7 * len(name), 16], fill=(0, 0, 0)); d.text((4, 2), name, fill=(255, 255, 255))
                row.append(im)
        if row:
            rows.append((Path(c["image"]).name, row))
    if not rows:
        raise SystemExit("nothing rendered")
    h = max(im.height for _, r in rows for im in r)
    cols = max(len(r) for _, r in rows)
    sheet = Image.new("RGB", ((a.width + 8) * cols + 8, (h + 22) * len(rows) + 8), (235, 235, 235))
    d = ImageDraw.Draw(sheet)
    for i, (n, r) in enumerate(rows):
        y = 8 + i * (h + 22)
        d.text((8, y), f"{n} moved {int(a.closer * 100)} % closer", fill=(0, 0, 0))
        for j, im in enumerate(r):
            sheet.paste(im, (8 + j * (a.width + 8), y + 14))
    sheet.save(a.out / "novel_sheet.jpg", quality=88)
    print(f"sheet: {a.out / 'novel_sheet.jpg'} ({len(rows)} views × {cols} models)")


if __name__ == "__main__":
    main()
