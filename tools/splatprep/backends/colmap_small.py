"""Smaller copies of large photos for COLMAP's feature extractor, and putting its results back at full size.

COLMAP's ``feature_extractor`` reads photos ahead of the SIFT threads and only shrinks each one to
``max_image_size`` when it is extracted. With 12 MP stills (431 × 4032×3024 HEIC) it held ~226 decoded photos at
once — 14 GB of ~60 MB allocations — and ran a 24 GB Mac out of memory into a kernel panic. Handing it copies at the
size it extracts at anyway keeps it to a few hundred MB, and gives the same features.

The copies are JPEGs named ``<name>.jpg`` (EXIF kept, so COLMAP still reads the focal length). Afterwards the
database is put back as if COLMAP had read the originals: image names, keypoint positions and scales, and camera
size and intrinsics (focal lengths and principal point; distortion terms are unitless).
"""
from __future__ import annotations

import os
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# make copies only when the photos are clearly bigger than COLMAP's working size
SMALL_COPY_FACTOR = 1.15
# COLMAP CPU SIFT memory per thread at 1600 px (measured: 4 threads 2.2 GB, 12 threads 6.5 GB), scaling with
# the pixel count; extraction may use this share of the computer's memory
SIFT_GB_PER_THREAD_1600 = 0.55
EXTRACTION_MEMORY_SHARE = 0.2

# COLMAP camera model id → number of focal-length parameters (then cx, cy)
_FOCALS = {0: 1, 1: 2, 2: 1, 3: 1, 4: 2, 5: 2, 6: 2, 7: 2, 8: 1, 9: 1, 10: 2}


@dataclass
class SmallCopies:
    image_dir: Path
    mask_dir: Optional[Path]
    names: Dict[str, str] = field(default_factory=dict)                  # original name → copy name
    scales: Dict[str, Tuple[float, float]] = field(default_factory=dict)  # original name → (sx, sy) copy → original
    sizes: Dict[str, Tuple[int, int]] = field(default_factory=dict)       # original name → original (w, h)
    max_side: int = 0                                                     # longest side of the copies


def _raw_size(path: Path) -> Tuple[int, int]:
    from PIL import Image

    from ..images import IMAGE_EXTS  # noqa: F401  (registers the HEIC opener)
    with Image.open(path) as im:
        return im.size            # stored orientation: COLMAP doesn't apply EXIF rotation either


def small_copies(image_dir: Path, names: Sequence[str], masks_dir: Optional[Path], out: Path,
                 resize_max: int) -> Optional[SmallCopies]:
    """Copies at ``resize_max`` (longest side) when any photo is bigger than ``SMALL_COPY_FACTOR`` × that; else None."""
    if resize_max <= 0 or not names:
        return None
    sizes = {n: _raw_size(Path(image_dir) / n) for n in names}
    if max(max(w, h) for w, h in sizes.values()) <= resize_max * SMALL_COPY_FACTOR:
        return None
    from PIL import Image

    img_out = out / "images"
    mask_out = out / "masks" if masks_dir is not None else None
    res = SmallCopies(img_out, mask_out)

    def one(name: str) -> Tuple[str, str, Tuple[float, float]]:
        w, h = sizes[name]
        s = min(1.0, resize_max / max(w, h))
        nw, nh = max(1, round(w * s)), max(1, round(h * s))
        copy = name + ".jpg"
        dst = img_out / copy
        dst.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(Path(image_dir) / name) as im:
            exif = im.info.get("exif")
            small = im.convert("RGB").resize((nw, nh), Image.LANCZOS) if (nw, nh) != (w, h) else im.convert("RGB")
            small.save(dst, "JPEG", quality=95, **({"exif": exif} if exif else {}))
        if mask_out is not None:
            m = Path(masks_dir) / (name + ".png")
            if m.is_file():
                md = mask_out / (copy + ".png")
                md.parent.mkdir(parents=True, exist_ok=True)
                with Image.open(m) as mi:
                    mi.resize((nw, nh), Image.NEAREST).save(md)
        return name, copy, (w / nw, h / nh)

    # a few threads, one photo each: memory stays at a handful of decoded photos
    with ThreadPoolExecutor(max_workers=max(1, min(6, (os.cpu_count() or 4) // 2))) as ex:
        for name, copy, sc in ex.map(one, list(names)):
            res.names[name] = copy
            res.scales[name] = sc
            res.sizes[name] = sizes[name]
    res.max_side = resize_max
    return res


def max_side(image_dir: Path, names: Sequence[str], sample: int = 20) -> int:
    """Longest side over (a sample of) the photos."""
    step = max(1, len(names) // sample)
    return max((max(_raw_size(Path(image_dir) / n)) for n in list(names)[::step]), default=0)


def extraction_threads(side: int, total_bytes: Optional[int] = None, cpus: Optional[int] = None) -> int:
    """Threads for COLMAP's feature extractor so it stays within EXTRACTION_MEMORY_SHARE of the memory."""
    cpus = cpus or os.cpu_count() or 4
    if total_bytes is None:
        try:
            total_bytes = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
        except (ValueError, OSError, AttributeError):
            return cpus
    per = SIFT_GB_PER_THREAD_1600 * max(1.0, (max(side, 1) / 1600) ** 2) * (1 << 30)
    return int(max(1, min(cpus, (total_bytes * EXTRACTION_MEMORY_SHARE) // per)))


def restore_full_size(database: Path, small: SmallCopies) -> None:
    """Puts names, keypoints and cameras in ``database`` back at the original photos' size."""
    back = {copy: name for name, copy in small.names.items()}
    con = sqlite3.connect(str(database))
    try:
        cams_done = set()
        rows = con.execute("SELECT image_id, name, camera_id FROM images").fetchall()
        for image_id, name, camera_id in rows:
            orig = back.get(name)
            if orig is None:
                continue
            sx, sy = small.scales[orig]
            con.execute("UPDATE images SET name=? WHERE image_id=?", (orig, image_id))
            kp = con.execute("SELECT rows, cols, data FROM keypoints WHERE image_id=?", (image_id,)).fetchone()
            if kp and kp[0] > 0 and kp[2]:
                r, c = int(kp[0]), int(kp[1])
                a = np.frombuffer(kp[2], dtype=np.float32).reshape(r, c).copy()
                a[:, 0] *= sx
                a[:, 1] *= sy
                if c == 4:
                    a[:, 2] *= (sx + sy) / 2                      # x, y, scale, orientation
                elif c == 6:
                    a[:, 2] *= sx; a[:, 3] *= sx; a[:, 4] *= sy; a[:, 5] *= sy   # x, y, affine a11 a12 a21 a22
                con.execute("UPDATE keypoints SET data=? WHERE image_id=?", (a.astype(np.float32).tobytes(), image_id))
            if camera_id in cams_done:
                continue
            cams_done.add(camera_id)
            cam = con.execute("SELECT model, params FROM cameras WHERE camera_id=?", (camera_id,)).fetchone()
            if not cam:
                continue
            model, blob = int(cam[0]), cam[1]
            p = np.frombuffer(blob, dtype=np.float64).copy()
            nf = _FOCALS.get(model, 1)
            if nf == 1:
                p[0] *= (sx + sy) / 2
            else:
                p[0] *= sx
                p[1] *= sy
            p[nf] *= sx
            p[nf + 1] *= sy
            w, h = small.sizes[orig]
            con.execute("UPDATE cameras SET width=?, height=?, params=? WHERE camera_id=?", (w, h, p.tobytes(), camera_id))
        con.commit()
    finally:
        con.close()

