"""No GPU for processing: the camera solve uses SIFT on the CPU unless the user chose a model (runner.cpu_features)."""
import types

import pytest

from splatprep import runner


@pytest.fixture
def backends(monkeypatch):
    learned = types.SimpleNamespace(python_modules=("torch", "lightglue"), display_name="SuperPoint + LightGlue",
                                    require=lambda ctx: None)
    sift = types.SimpleNamespace(python_modules=("cv2",), display_name="OpenCV SIFT + ratio test",
                                 require=lambda ctx: None)
    csift = types.SimpleNamespace(python_modules=("pycolmap",), display_name="COLMAP SIFT", require=lambda ctx: None)
    table = {"superpoint_lightglue": learned, "aliked_lightglue": learned, "opencv_sift": sift, "colmap_sift": csift}
    monkeypatch.setattr(runner, "get_backend", lambda m: table[m])
    monkeypatch.setenv("SPLATPREP_GPU_NAME", "AMD Custom GPU 0405")
    return table


def test_default_becomes_sift_on_the_cpu(backends):
    model, note = runner.cpu_features({"features": "superpoint_lightglue"}, "cpu", 200)
    assert model == runner.CPU_FEATURES
    assert note == "No GPU for processing (PyTorch can't use the AMD Custom GPU 0405) — solving on the CPU with COLMAP SIFT"
    assert runner.cpu_features({"features": "auto"}, "cpu", 200)[0] == runner.CPU_FEATURES


def test_explicit_choice_is_kept_with_a_time_warning(backends):
    model, note = runner.cpu_features({"features": "superpoint_lightglue", "keepFeatures": True}, "cpu", 300)
    assert model == "superpoint_lightglue" and "roughly 50 min for 300 photos" in note
    model, note = runner.cpu_features({"features": "aliked_lightglue"}, "cpu", 60)
    assert model == "aliked_lightglue" and "roughly 10 min" in note
    assert runner.cpu_features({"features": "colmap_sift"}, "cpu", 60) == ("colmap_sift", None)


def test_a_gpu_changes_nothing(backends):
    assert runner.cpu_features({"features": "superpoint_lightglue"}, "cuda", 200) == ("superpoint_lightglue", None)
    assert runner.cpu_features({"features": "auto"}, "cuda", 200) == ("superpoint_lightglue", None)


def test_fallbacks_when_sift_is_missing(backends, monkeypatch):
    def broken(ctx):
        raise RuntimeError("missing")
    backends["colmap_sift"].require = broken
    model, note = runner.cpu_features({"features": "superpoint_lightglue"}, "cpu", 10)
    assert model == "opencv_sift" and note.endswith("with OpenCV SIFT")
    backends["opencv_sift"].require = broken
    assert runner.cpu_features({"features": "superpoint_lightglue"}, "cpu", 10) == ("superpoint_lightglue", None)
