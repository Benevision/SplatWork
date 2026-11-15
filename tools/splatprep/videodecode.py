"""Opening videos for frame extraction / home-video restoration, decoded on the GPU's video engine when it helps.

PyAV's wheel (FFmpeg with NVDEC / CUVID) can decode H.264, HEVC and AV1 on an NVIDIA GPU's video decoder — a block
of its own, separate from the cores training uses — and so leave the CPU free while another job trains. It has no
VA-API, so AMD / Intel GPUs (an integrated Radeon) decode on the CPU.

``SPLATPREP_VIDEO_DECODE``: "auto" (default: NVDEC when it works for this video), "cpu", or "cuda" (NVDEC when it
works; the same checks as auto). It is used only when it gives the *same pictures*: once per codec and size,
the first PROBE_FRAMES frames are decoded both ways — the grey images (what frame picking measures sharpness and
motion on) must be identical and the colour ones within RGB_TOLERANCE levels. Any error, or a difference, means
the CPU, silently apart from a log line. The decoded frames come back in system memory either way, so the code
that uses them doesn't change.
"""
from __future__ import annotations

import os
import threading
import time
from pathlib import Path
from typing import Callable, Dict, Tuple

MODE_ENV = "SPLATPREP_VIDEO_DECODE"
PROBE_FRAMES = 6
RGB_TOLERANCE = 2

_lock = threading.Lock()
_probed: Dict[Tuple, Tuple[bool, str]] = {}     # (path-independent key) → (use NVDEC, why)


def mode() -> str:
    m = os.environ.get(MODE_ENV, "auto").strip().lower()
    return m if m in ("auto", "cpu", "cuda") else "auto"


def _hwaccel(av):
    from av.codec.hwaccel import HWAccel

    return HWAccel(device_type="cuda", allow_software_fallback=False)


def _first_frames(av, path: Path, hwaccel, n: int):
    """(grey, rgb24, was it decoded on the hardware) of the first n frames."""
    import numpy as np

    c = av.open(str(path), hwaccel=hwaccel) if hwaccel is not None else av.open(str(path))
    try:
        s = c.streams.video[0]
        out = []
        for f in c.decode(s):
            out.append((np.asarray(f.reformat(format="gray").to_ndarray()),
                        np.asarray(f.to_ndarray(format="rgb24"))))
            if len(out) >= n:
                break
        hw = bool(getattr(s.codec_context, "is_hwaccel", False))
        return out, hw
    finally:
        c.close()


def probe(av, path: Path) -> Tuple[bool, str]:
    """Whether NVDEC decodes this video's codec / size the same as the CPU (cached per codec and size)."""
    import numpy as np

    try:
        from av.codec.hwaccel import hwdevices_available

        if "cuda" not in hwdevices_available():
            return False, "no CUDA video decoder in this FFmpeg"
    except Exception as e:  # noqa: BLE001 - PyAV < 14
        return False, f"this PyAV can't decode on the GPU ({type(e).__name__})"
    try:
        c = av.open(str(path))
        try:
            cc = c.streams.video[0].codec_context
            key = (cc.name, cc.width, cc.height, str(getattr(cc, "format", None) and cc.format.name))
        finally:
            c.close()
    except Exception as e:  # noqa: BLE001
        return False, f"can't read the video ({e})"
    with _lock:
        if key in _probed:
            return _probed[key]
    try:
        cpu, _ = _first_frames(av, path, None, PROBE_FRAMES)
        gpu, hw = _first_frames(av, path, _hwaccel(av), PROBE_FRAMES)
        if not hw:
            res = (False, f"NVDEC doesn't decode {key[0]} here")
        elif len(gpu) != len(cpu) or not cpu:
            res = (False, f"NVDEC gave {len(gpu)} frames, the CPU {len(cpu)}")
        else:
            grey_same = all(np.array_equal(a[0], b[0]) for a, b in zip(cpu, gpu))
            rgb_diff = max(int(np.abs(a[1].astype(int) - b[1].astype(int)).max()) for a, b in zip(cpu, gpu))
            if not grey_same:
                res = (False, "NVDEC's pictures differ from the CPU's (grey)")
            elif rgb_diff > RGB_TOLERANCE:
                res = (False, f"NVDEC's colours differ from the CPU's by up to {rgb_diff} levels")
            else:
                res = (True, f"checked against the CPU on {len(cpu)} frames: grey identical, colour within "
                             f"{rgb_diff} level(s)")
    except Exception as e:  # noqa: BLE001 - no NVIDIA driver, unsupported codec, …
        res = (False, f"NVDEC unavailable ({type(e).__name__}: {str(e)[:120]})")
    with _lock:
        _probed[key] = res
    return res


class Opened:
    """The container plus how it is decoded; ``done(frames)`` logs the decode rate."""

    def __init__(self, container, decoder: str, log: Callable[[str], None]):
        self.container = container
        self.decoder = decoder
        self.log = log
        self.t0 = time.perf_counter()

    def done(self, frames: int) -> None:
        secs = time.perf_counter() - self.t0
        if frames and secs > 0:
            self.log(f"video decoding: {frames} frames in {secs:.1f} s ({frames / secs:.0f} fps) on {self.decoder}")


def open_video(path: Path, log: Callable[[str], None] = lambda m: None) -> Opened:
    """``av.open(path)``, with NVDEC when the setting allows it and the check passes (see the module doc)."""
    try:
        import av
    except ImportError:
        raise
    m = mode()
    if m == "cpu":
        log("video decoding: CPU (Decode video on: CPU)")
        return Opened(av.open(str(path)), "the CPU", log)
    use, why = probe(av, Path(path))
    if use:
        try:
            c = av.open(str(path), hwaccel=_hwaccel(av))
            log(f"video decoding: NVDEC on the NVIDIA GPU's video engine ({why})")
            return Opened(c, "NVDEC", log)
        except Exception as e:  # noqa: BLE001
            why = f"NVDEC failed to open ({e})"
    log(f"video decoding: CPU ({why})")
    return Opened(av.open(str(path)), "the CPU", log)
