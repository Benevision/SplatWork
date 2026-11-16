"""New Project → Process on a worker (studio/remote_prep.py, as the Mac's "Process on"): the camera solve runs on the
chosen worker with the photos sent once; a step the worker can't do (a model it hasn't got) runs on this PC; a
worker that isn't there means this PC. Two real worker servers (local and remote) with fake backends and the
fake splatprep."""
from __future__ import annotations

import json
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

pytest.importorskip("uvicorn")
pytest.importorskip("PIL")

from test_studio_compute import pipeline, workers  # noqa: E402,F401

from splatworker.studio import compute, remote_prep  # noqa: E402
from splatworker.studio import devices as D  # noqa: E402
from splatworker.studio.client import Client  # noqa: E402
from splatworker.studio.presets import ProjectOptions  # noqa: E402
from splatworker.studio.projects import Project  # noqa: E402


def photos_project(tmp_path, process_on, masks=()):
    from PIL import Image

    src = tmp_path / "photos"
    src.mkdir()
    for i in range(6):
        Image.fromarray(np.full((48, 64, 3), 40 * i, np.uint8)).save(src / f"p{i}.jpg")
    opts = ProjectOptions(name="Garden", source_kind="photos", source=str(src), preset="preview", sky=False,
                          mask_models=list(masks), process_on=process_on, blur_rejection=False)
    proj = Project.create(tmp_path / "projects", "Garden", opts.to_dict())
    return proj, opts


def test_camera_solve_on_the_worker_and_masks_it_lacks_here(tmp_path, workers, fake_splatprep):  # noqa: F811
    lport, lback, _ = workers("local", local=True)
    rport, rback, _ = workers("remote", token="r")
    reg = D.DeviceRegistry(tmp_path / "devices.json")
    dev = reg.add_manual("127.0.0.1", rport, token="r")
    assert dev.online and "process" in dev.capabilities
    proj, opts = photos_project(tmp_path, dev.id, masks=["person"])     # the fake worker has no "person" model
    p, events = pipeline(lport, proj, opts, reg)
    p.train_enabled = False
    p._titles = [t for t in p._titles if t != "Training"]
    p.run()
    assert ("done", {"output": None, "dataset": proj.dataset_dir}) in events or any(k == "done" for k, _ in events), \
        [e for e in events if e[0] in ("failed", "log")][-5:]
    log = (proj.root / "prep.log").read_text()
    assert f"processing on {dev.name}" in log
    assert f"{dev.name} doesn't have person — masks (person) on this PC" in log
    local_tasks = [c.spec.get("task") for c in lback.contexts if c.kind == "process"]
    remote_tasks = [c.spec.get("task") for c in rback.contexts if c.kind == "process"]
    assert local_tasks == ["masks"] and remote_tasks == ["sfm"]
    assert (proj.dataset_dir / "dataset.json").is_file()
    assert json.loads((proj.root / "project.json").read_text())["processedOn"] == dev.name
    # the photos stay on the worker, recorded for training there (no second upload)
    rec = json.loads((proj.dataset_dir / compute.UPLOADS_FILE).read_text())
    remote = Client(dev.base_url, "r")
    assert rec[dev.base_url]["datasetId"] in {d["datasetId"] for d in remote.datasets()}
    assert all(k.startswith("images/") for k in rec[dev.base_url]["files"])


def test_offline_worker_means_this_pc(tmp_path, workers, fake_splatprep, monkeypatch):  # noqa: F811
    lport, lback, _ = workers("local", local=True)
    reg = D.DeviceRegistry(tmp_path / "devices.json")
    dev = reg.add_manual("127.0.0.1", 9, token="r", check=False)                # nothing listens there
    monkeypatch.setattr(remote_prep, "FIND_SECONDS", 0.2)
    proj, opts = photos_project(tmp_path, dev.id)
    p, events = pipeline(lport, proj, opts, reg)
    p.train_enabled = False
    p._titles = [t for t in p._titles if t != "Training"]
    p.run()
    log = (proj.root / "prep.log").read_text()
    assert "isn't on the network — preparing the dataset on this PC" in log
    assert [c.spec.get("task") for c in lback.contexts if c.kind == "process"] == ["sfm"]


def test_missing_models_and_merge(tmp_path):
    info = {"models": ["superpoint_lightglue", "sky"]}
    assert remote_prep.missing_models(info, ["person", "none"], True, "superpoint_lightglue") == ["person"]
    assert remote_prep.missing_models({}, ["person"], True, "x") == []          # an old worker: unknown
    z = tmp_path / "o.zip"
    with zipfile.ZipFile(z, "w") as zf:
        zf.writestr("masks/a.png", b"new")
        zf.writestr("dataset.json", b"{}")
    dest = tmp_path / "d"
    (dest / "masks").mkdir(parents=True)
    (dest / "masks" / "a.png").write_bytes(b"old")
    (dest / "keep.txt").write_text("k")
    assert remote_prep.merge_zip(z, dest) == 2
    assert (dest / "masks" / "a.png").read_bytes() == b"new" and (dest / "keep.txt").exists()
    bad = tmp_path / "bad.zip"
    with zipfile.ZipFile(bad, "w") as zf:
        zf.writestr("../x", b"x")
    with pytest.raises(RuntimeError, match="unsafe"):
        remote_prep.merge_zip(bad, dest)


def test_choices_and_slots():
    class R:
        def __init__(self, devs):
            self.devs = devs

        def devices(self):
            return self.devs
    on = D.Device(id="a", name="BensPC", host="h", port=1, status="online",
                  info={"capabilities": ["train", "process"], "workerVersion": "1.7.10",
                        "slots": [{"id": "cuda:0", "kinds": ["train", "process", "census"]},
                                  {"id": "vk:x", "kinds": ["train"], "slow": True}], "gpus": [{"name": "RTX"}]})
    trains = D.Device(id="b", name="Box", host="h", port=2, status="online", info={"capabilities": ["train"]})
    off = D.Device(id="c", name="Old", host="h", port=3, status="offline")
    from splatworker.gui.gtk.studio import devices_page  # noqa: F401 - needs GTK

    assert devices_page.process_on_choices(R([on, trains, off])) == [
        ("local", "This PC"), ("a", "BensPC"), ("b", "Box (can't process)"), ("c", "Old (offline)")]
    assert on.slots == 2                                          # the 610M counts from 1.7.8
    on.info["workerVersion"] = "1.7.7"
    assert on.slots == 1
