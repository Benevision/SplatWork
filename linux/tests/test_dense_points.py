"""Dense points on the worker: request validation, /v1/info "dense", the Linux app's option."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from splatworker import processing  # noqa: E402
from splatworker.processing import ProcessingError, validate_request  # noqa: E402


def test_dense_requests_validate():
    r = validate_request({"task": "sfm", "dense": {"method": "auto"}})
    assert r["dense"] == {"method": "auto"}
    assert validate_request({"task": "sfm", "dense": True})["dense"] is True
    with pytest.raises(ProcessingError):
        validate_request({"task": "sfm", "dense": "yes"})
    assert validate_request({"task": "dense", "datasetDir": "sfm"})["datasetDir"] == "sfm"
    with pytest.raises(ProcessingError):
        validate_request({"task": "dense", "datasetDir": "../elsewhere"})
    with pytest.raises(ProcessingError):
        validate_request({"task": "dense", "datasetDir": "/abs"})


def test_dense_methods_cached(monkeypatch):
    processing._dense_cache.clear()
    calls = []
    monkeypatch.setattr(processing, "splatprep_available", lambda: calls.append(1) or False)
    assert processing.dense_methods() is None and processing.dense_methods() is None
    assert len(calls) == 1
    processing._dense_cache.clear()


def test_linux_app_option_adds_dense_to_the_solve():
    from splatworker.studio.presets import ProjectOptions

    o = ProjectOptions(name="x", source_kind="photos", source="/tmp/p")
    assert o.dense_points is False and "dense" not in o.sfm_request(False, False, False)
    o.dense_points = True
    assert o.sfm_request(False, False, False)["dense"] == {"method": "auto"}
    assert ProjectOptions.from_dict(o.to_dict()).dense_points is True


def test_whole_pipeline_job_keeps_dense_in_its_solve():
    from splatworker.pipeline import sfm_request

    fs = {"frames": [{"image": "images/a.jpg", "width": 10, "height": 10}]}
    r, _ = sfm_request({"features": "colmap_sift", "dense": {"method": "auto", "maxPoints": 200000}}, fs, False)
    assert r["dense"] == {"method": "auto", "maxPoints": 200000} and r["task"] == "sfm"
