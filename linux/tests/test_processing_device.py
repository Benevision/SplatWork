"""Where processing runs (/v1/info "processing", setup wording), and what other computers see of a failed job of
the desktop app."""
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def test_processing_device(monkeypatch):
    from splatworker import gpu, gpu_vendors

    deck = types.SimpleNamespace(vendor="amd", name="AMD Custom GPU 0405", integrated=True, memory_mb=1024)
    monkeypatch.setattr(gpu_vendors, "scan", lambda root: [deck])
    monkeypatch.setattr(gpu, "torch_backend", lambda: "cpu")
    monkeypatch.setattr(gpu.importlib.util, "find_spec", lambda name: object())
    p = gpu.processing_device(refresh=True)
    assert p["device"] == "cpu" and p["reason"] == "PyTorch can't use the AMD Custom GPU 0405" and p["gpu"] is None
    monkeypatch.setattr(gpu, "torch_backend", lambda: "rocm")
    p = gpu.processing_device(refresh=True)
    assert p == {"device": "rocm", "gpu": "AMD Custom GPU 0405", "reason": None, "gpuName": "AMD Custom GPU 0405"}
    gpu._processing_cache.clear()


def test_setup_says_what_the_cpu_does():
    from splatworker.gui.runtime_plan import cpu_processing_note

    deck = types.SimpleNamespace(vendor="amd", name="AMD Custom GPU 0405")
    note = cpu_processing_note(deck)
    assert "PyTorch can't use the AMD Custom GPU 0405" in note and "COLMAP SIFT" in note and "ROCm" in note
    assert "ROCm" not in cpu_processing_note(types.SimpleNamespace(vendor="intel", name="Intel Iris Xe"))


def test_explicit_feature_choice_is_sent():
    from splatworker.studio.presets import ProjectOptions

    o = ProjectOptions(name="x", source_kind="photos", source="p")
    assert "keepFeatures" not in o.sfm_request(False, False, False)
    o.features_explicit = True
    assert o.sfm_request(False, False, False)["keepFeatures"] is True


def test_other_computers_see_where_a_local_failure_is_explained(tmp_path):
    pytest.importorskip("fastapi")
    from conftest import FakeBackend
    from fastapi.testclient import TestClient

    from splatworker.config import ServerSettings
    from splatworker.server import create_app

    settings = ServerSettings(data_dir=tmp_path, mdns=False, allow_local_datasets=True, local_secret="k" * 32)
    app = create_app(settings, backend=FakeBackend(), devices=[None])
    with TestClient(app) as c:
        m = app.state.manager
        job = m.submit("process", "d1", "Deck project", "request", {"task": "sfm"}, total_steps=1000,
                       origin="local")
        with m.cond:
            m.queue.remove(job.id)
        job.update(persist=True, state="failed", message="sfm: only 3 of 180 photos placed (project Garden)")
        s = c.get(f"/v1/jobs/{job.id}").json()             # another computer (no local secret)
        assert s["message"].startswith("failed on that PC: the reason is on its Jobs page") and "Garden" not in s["message"]
        r = c.get(f"/v1/jobs/{job.id}/log")
        assert r.status_code == 403 and "prep.log" in r.json()["error"]
