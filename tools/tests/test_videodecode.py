"""Video decoding on NVDEC when it gives the same pictures (videodecode.py), with a fake PyAV."""
import sys
import types

import numpy as np
import pytest

from splatprep import videodecode as V


class _Fmt:
    name = "yuv420p"


class _Frame:
    def __init__(self, grey, rgb):
        self.grey, self.rgb = grey, rgb

    def reformat(self, format=None, **_):
        return types.SimpleNamespace(to_ndarray=lambda: self.grey)

    def to_ndarray(self, format=None):
        return self.rgb


def fake_av(cuda=True, hw_works=True, grey_shift=0, rgb_shift=0, open_fails=False):
    """A PyAV whose hardware decode returns the CPU frames shifted by the given amounts."""
    rng = np.random.default_rng(0)
    frames = [(rng.integers(0, 250, (8, 8), dtype=np.uint8), rng.integers(0, 250, (8, 8, 3), dtype=np.uint8))
              for _ in range(10)]
    opened = []

    def open_(path, hwaccel=None):
        if hwaccel is not None and open_fails:
            raise OSError("no NVIDIA driver")
        hw = hwaccel is not None and hw_works
        opened.append("hw" if hwaccel is not None else "cpu")
        cc = types.SimpleNamespace(name="h264", width=8, height=8, format=_Fmt(), is_hwaccel=hw)
        stream = types.SimpleNamespace(codec_context=cc)

        def decode(_s):
            for g, c in frames:
                yield _Frame(g + (grey_shift if hw else 0), c + (rgb_shift if hw else 0))
        return types.SimpleNamespace(streams=types.SimpleNamespace(video=[stream]), decode=decode,
                                     close=lambda: None)

    av = types.ModuleType("av")
    av.open = open_
    hwmod = types.ModuleType("av.codec.hwaccel")
    hwmod.HWAccel = lambda device_type, allow_software_fallback=True: ("hw", device_type)
    hwmod.hwdevices_available = lambda: ["cuda", "vaapi"] if cuda else ["vaapi"]
    codec = types.ModuleType("av.codec")
    codec.hwaccel = hwmod
    av.codec = codec
    return av, {"av": av, "av.codec": codec, "av.codec.hwaccel": hwmod}, opened


@pytest.fixture
def use(monkeypatch):
    def install(**kw):
        av, mods, opened = fake_av(**kw)
        for k, m in mods.items():
            monkeypatch.setitem(sys.modules, k, m)
        V._probed.clear()
        return opened
    yield install
    V._probed.clear()


def test_identical_pictures_use_nvdec(use, monkeypatch, tmp_path):
    opened = use()
    logs = []
    o = V.open_video(tmp_path / "v.mp4", logs.append)
    assert o.decoder == "NVDEC" and opened[-1] == "hw"
    assert "NVDEC" in logs[0] and "grey identical" in logs[0]
    o.done(0)
    V.open_video(tmp_path / "v.mp4", logs.append)          # checked once per codec and size
    # checked once: a CPU decode only in the first call's check; each call reads the codec / size (one CPU open)
    assert opened == ["cpu", "cpu", "hw", "hw", "cpu", "hw"]


@pytest.mark.parametrize("kw, why", [
    ({"cuda": False}, "no CUDA video decoder"),
    ({"hw_works": False}, "doesn't decode h264"),
    ({"grey_shift": 1}, "differ from the CPU's (grey)"),
    ({"rgb_shift": 3}, "colours differ from the CPU's by up to 3"),
    ({"open_fails": True}, "NVDEC unavailable (OSError: no NVIDIA driver)"),
])
def test_any_difference_or_error_means_the_cpu(use, tmp_path, kw, why):
    use(**kw)
    logs = []
    o = V.open_video(tmp_path / "v.mp4", logs.append)
    assert o.decoder == "the CPU" and why in logs[0], logs


def test_small_colour_rounding_is_accepted(use, tmp_path):
    use(rgb_shift=2)
    assert V.open_video(tmp_path / "v.mp4").decoder == "NVDEC"


def test_setting_cpu(use, monkeypatch, tmp_path):
    opened = use()
    monkeypatch.setenv(V.MODE_ENV, "cpu")
    logs = []
    assert V.open_video(tmp_path / "v.mp4", logs.append).decoder == "the CPU"
    assert opened == ["cpu"] and logs == ["video decoding: CPU (Decode video on: CPU)"]
    monkeypatch.setenv(V.MODE_ENV, "bogus")
    assert V.mode() == "auto"


def test_rate_is_logged(use, tmp_path):
    use()
    logs = []
    o = V.open_video(tmp_path / "v.mp4", logs.append)
    o.done(240)
    assert logs[-1].startswith("video decoding: 240 frames in ") and logs[-1].endswith("on NVDEC")
