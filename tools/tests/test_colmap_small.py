"""Small copies for COLMAP's feature extractor and the memory guard (the 2026-11-16 out-of-memory kernel panic)."""
import sqlite3
import sys

import numpy as np
import pytest
from PIL import Image

from splatprep.backends.colmap_small import extraction_threads, restore_full_size, small_copies
from splatprep.errors import SplatPrepError
from splatprep.mapper import run_logged


def test_small_copies_skip_small_photos(tmp_path):
    (tmp_path / "a.jpg").write_bytes(b"")
    Image.new("RGB", (1600, 1200)).save(tmp_path / "a.jpg")
    assert small_copies(tmp_path, ["a.jpg"], None, tmp_path / "out", 1600) is None


def test_small_copies_and_restore(tmp_path):
    img, masks = tmp_path / "images", tmp_path / "masks"
    img.mkdir(); masks.mkdir()
    Image.new("RGB", (4032, 3024), (90, 120, 30)).save(img / "f1.jpg")
    Image.new("L", (4032, 3024), 255).save(masks / "f1.jpg.png")
    s = small_copies(img, ["f1.jpg"], masks, tmp_path / "small", 1600)
    assert s is not None and s.names["f1.jpg"] == "f1.jpg.jpg"
    with Image.open(s.image_dir / "f1.jpg.jpg") as im:
        assert im.size == (1600, 1200)
    with Image.open(s.mask_dir / "f1.jpg.jpg.png") as m:
        assert m.size == (1600, 1200)
    sx, sy = s.scales["f1.jpg"]
    assert sx == pytest.approx(4032 / 1600) and sy == pytest.approx(3024 / 1200)

    db = tmp_path / "database.db"
    con = sqlite3.connect(db)
    con.execute("CREATE TABLE cameras (camera_id INTEGER PRIMARY KEY, model INTEGER, width INTEGER, height INTEGER, params BLOB, prior_focal_length INTEGER)")
    con.execute("CREATE TABLE images (image_id INTEGER PRIMARY KEY, name TEXT, camera_id INTEGER)")
    con.execute("CREATE TABLE keypoints (image_id INTEGER PRIMARY KEY, rows INTEGER, cols INTEGER, data BLOB)")
    con.execute("INSERT INTO cameras VALUES (1, 2, 1600, 1200, ?, 1)", (np.array([1300.0, 800.0, 600.0, 0.01]).tobytes(),))
    con.execute("INSERT INTO images VALUES (1, 'f1.jpg.jpg', 1)")
    kp = np.array([[800.0, 600.0, 2.0, 0.0, 0.0, 2.0]], np.float32)
    con.execute("INSERT INTO keypoints VALUES (1, 1, 6, ?)", (kp.tobytes(),))
    con.commit(); con.close()

    restore_full_size(db, s)
    con = sqlite3.connect(db)
    assert con.execute("SELECT name FROM images").fetchone()[0] == "f1.jpg"
    w, h, params = con.execute("SELECT width, height, params FROM cameras").fetchone()
    p = np.frombuffer(params, np.float64)
    assert (w, h) == (4032, 3024)
    assert p[1] == pytest.approx(800 * sx) and p[2] == pytest.approx(600 * sy) and p[3] == pytest.approx(0.01)
    k = np.frombuffer(con.execute("SELECT data FROM keypoints").fetchone()[0], np.float32).reshape(1, 6)
    assert k[0, 0] == pytest.approx(800 * sx, rel=1e-5) and k[0, 1] == pytest.approx(600 * sy, rel=1e-5)
    assert k[0, 2] == pytest.approx(2 * sx, rel=1e-5) and k[0, 5] == pytest.approx(2 * sy, rel=1e-5)


@pytest.mark.skipif(sys.platform == "win32", reason="the guard reads memory with ps / /proc")
def test_memory_guard_stops_a_runaway_child(tmp_path, monkeypatch):
    monkeypatch.setenv("SPLATPREP_CHILD_MEMORY_FRACTION", "0.005")     # ~100–500 MB on a test machine
    hog = "import time\nb=[]\nfor _ in range(400):\n    b.append(bytearray(50*1024*1024))\n    time.sleep(0.05)\n"
    with pytest.raises(SplatPrepError, match="was stopped"):
        run_logged([sys.executable, "-c", hog], tmp_path / "log.txt")


def test_extraction_threads_follow_memory():
    gb = 1 << 30
    assert extraction_threads(1600, 24 * gb, 12) == 8        # ~4.4 GB on a 24 GB Mac (was 12 threads, 6.5 GB)
    assert extraction_threads(1600, 96 * gb, 32) == 32
    assert extraction_threads(3200, 8 * gb, 8) == 1
