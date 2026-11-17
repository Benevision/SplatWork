"""Real-photo benchmark for splatwork-vk-match (docs/dev/GPU_PROCESSING.md, stage 1).

  python3 match_bench.py extract PHOTOS OUT [--images 40] [--keypoints 4096] [--window 5]
      OpenCV SIFT on the first N photos (sorted) → OUT/<name>.sift (u32 n + n × 128 bytes) and OUT/pairs.txt
      (sequential pairs, each photo with the next `window`).
  python3 match_bench.py opencv OUT [--ratio 0.8]
      The CPU baseline: cv2.BFMatcher (L2) k = 2 both ways, ratio test, mutual check → OUT/opencv.jsonl, and the
      time per pair.
  python3 match_bench.py compare OUT VK.jsonl
      splatwork-vk-match's matches (its `match` output) against OpenCV's: identical pairs, and match counts.
  python3 match_bench.py run PHOTOS [--images 30] [--device N]
      All of it in a temporary folder: extract, OpenCV, splatwork-vk-match on the GPU, compare; prints one
      summary line (the Flatpak installs this file as /app/share/splatwork/match_bench.py).
"""
from __future__ import annotations

import argparse
import json
import struct
import time
from pathlib import Path

import numpy as np


def extract(args) -> None:
    import cv2

    photos = sorted(p for p in Path(args.photos).iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    photos = photos[: args.images]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    sift = cv2.SIFT_create(nfeatures=args.keypoints)
    t0 = time.perf_counter()
    names = []
    for p in photos:
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        _, d = sift.detectAndCompute(img, None)
        d = np.zeros((0, 128), np.float32) if d is None else d
        u8 = np.clip(np.round(d), 0, 255).astype(np.uint8)     # OpenCV's SIFT values are integers 0…255
        with open(out / f"{p.stem}.sift", "wb") as f:
            f.write(struct.pack("<I", len(u8)))
            f.write(u8.tobytes())
        names.append(p.stem)
    secs = time.perf_counter() - t0
    with open(out / "pairs.txt", "w") as f:
        for i, a in enumerate(names):
            for b in names[i + 1: i + 1 + args.window]:
                f.write(f"{a} {b}\n")
    print(json.dumps({"images": len(names), "extractSecondsPerImage": secs / max(len(names), 1)}))


def _load(out: Path, name: str) -> np.ndarray:
    raw = (out / f"{name}.sift").read_bytes()
    n = struct.unpack("<I", raw[:4])[0]
    return np.frombuffer(raw[4:], np.uint8).reshape(n, 128)


def opencv(args) -> None:
    import cv2

    out = Path(args.out)
    pairs = [line.split() for line in (out / "pairs.txt").read_text().splitlines() if line.strip()]
    cache = {}
    bf = cv2.BFMatcher(cv2.NORM_L2)
    t0 = time.perf_counter()
    rows = []
    for a, b in pairs:
        da = cache.setdefault(a, _load(out, a).astype(np.float32))
        db = cache.setdefault(b, _load(out, b).astype(np.float32))

        def nn(x, y):
            best = {}
            for m in bf.knnMatch(x, y, k=2):
                if not m:
                    continue
                ok = len(m) < 2 or m[0].distance < args.ratio * m[1].distance
                best[m[0].queryIdx] = (m[0].trainIdx, ok)
            return best
        ab, ba = nn(da, db), nn(db, da)
        ms = sorted([i, j] for i, (j, ok) in ab.items() if ok and ba.get(j, (None, False)) == (i, True))
        rows.append({"a": a, "b": b, "matches": ms})
    secs = time.perf_counter() - t0
    with open(out / "opencv.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(json.dumps({"device": f"OpenCV BFMatcher ({cv2.getNumThreads()} threads)", "pairs": len(pairs),
                      "msPerPair": 1e3 * secs / max(len(pairs), 1),
                      "matchesPerPair": sum(len(r["matches"]) for r in rows) / max(len(rows), 1)}))


def compare(args) -> dict:
    out = Path(args.out)
    ref = {(r["a"], r["b"]): r["matches"] for r in map(json.loads, (out / "opencv.jsonl").read_text().splitlines())}
    vk = [json.loads(line) for line in Path(args.vk).read_text().splitlines() if line.startswith("{")]
    pairs = [r for r in vk if r.get("type") == "pair"]
    same = sum(1 for r in pairs if sorted(map(list, r["matches"])) == ref.get((r["a"], r["b"])))
    diff = [(r["a"], r["b"], len(r["matches"]), len(ref.get((r["a"], r["b"]), []))) for r in pairs
            if sorted(map(list, r["matches"])) != ref.get((r["a"], r["b"]))]
    res = {"pairs": len(pairs), "identical": same, "different": diff[:5]}
    print(json.dumps(res))
    return res


def run(args) -> None:
    import contextlib
    import io
    import shutil
    import subprocess
    import tempfile

    exe = shutil.which("splatwork-vk-match") or "/app/bin/splatwork-vk-match"
    with tempfile.TemporaryDirectory(prefix="vk-match-bench-") as tmp:
        ns = argparse.Namespace(photos=args.photos, out=tmp, images=args.images, keypoints=4096, window=5, ratio=0.8)
        print(f"SIFT on {args.images} photos…", flush=True)
        with contextlib.redirect_stdout(io.StringIO()) as o:
            extract(ns)
        ext = json.loads(o.getvalue())
        print("OpenCV matching (CPU)…", flush=True)
        with contextlib.redirect_stdout(io.StringIO()) as o:
            opencv(ns)
        cv = json.loads(o.getvalue())
        print("Vulkan matching (GPU)…", flush=True)
        cmd = [exe, "match", "--desc", tmp, "--pairs", f"{tmp}/pairs.txt"] + (
            ["--device", str(args.device)] if args.device is not None else [])
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            raise SystemExit(f"splatwork-vk-match failed: {r.stderr.strip() or r.stdout.strip()[-300:]}")
        (Path(tmp) / "vk.jsonl").write_text(r.stdout)
        vk = json.loads(r.stdout.strip().splitlines()[-1])
        with contextlib.redirect_stdout(io.StringIO()):
            cmp_ = compare(argparse.Namespace(out=tmp, vk=f"{tmp}/vk.jsonl"))
    summary = {"type": "match-bench", "photos": ext["images"], "pairs": vk["pairs"], "gpu": vk["device"],
               "gpuMsPerPair": round(vk["msPerPair"], 2), "cpu": cv["device"], "cpuMsPerPair": round(cv["msPerPair"], 2),
               "speedup": round(cv["msPerPair"] / vk["msPerPair"], 1) if vk["msPerPair"] else None,
               "identicalPairs": f"{cmp_['identical']} of {cmp_['pairs']}",
               "matchesPerPair": round(cv["matchesPerPair"], 1)}
    print(f"GPU {summary['gpu']}: {summary['gpuMsPerPair']} ms per pair · CPU {summary['cpu']}: "
          f"{summary['cpuMsPerPair']} ms per pair · {summary['speedup']}× · identical {summary['identicalPairs']}")
    print(json.dumps(summary))


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("extract")
    e.add_argument("photos")
    e.add_argument("out")
    e.add_argument("--images", type=int, default=40)
    e.add_argument("--keypoints", type=int, default=4096)
    e.add_argument("--window", type=int, default=5)
    o = sub.add_parser("opencv")
    o.add_argument("out")
    o.add_argument("--ratio", type=float, default=0.8)
    c = sub.add_parser("compare")
    c.add_argument("out")
    c.add_argument("vk")
    r = sub.add_parser("run")
    r.add_argument("photos")
    r.add_argument("--images", type=int, default=30)
    r.add_argument("--device", type=int, default=None)
    a = ap.parse_args()
    {"extract": extract, "opencv": opencv, "compare": compare, "run": run}[a.cmd](a)


if __name__ == "__main__":
    main()
