"""GPU discovery.

``list_gpus`` lists the GPUs jobs can run on (their indices are the job slots). It uses NVML (``nvidia-ml-py``) when available because it does not create a CUDA
context in the server process; otherwise falls back to ``torch.cuda``. Both
are optional: without them the worker reports no GPUs.

The CLI sets ``CUDA_DEVICE_ORDER=PCI_BUS_ID`` so that torch indices match
NVML / ``nvidia-smi`` indices. A ROCm build of PyTorch answers through the same ``torch.cuda`` API (AMD GPUs,
``"vendor": "amd"``); Intel GPUs through ``torch.xpu`` are not job slots (nothing trains there yet).

``adapters`` lists every graphics card of any vendor from sysfs (``gpu_vendors.py``), for display and for
the ``/v1/info`` "adapters" field.
"""

from __future__ import annotations

import importlib.util
from typing import Dict, List, Optional


def normalize_pci(bus_id: str) -> str:
    """"00000000:01:00.0" (NVML) / "0000:01:00.0" (sysfs, Vulkan) → "0000:01:00.0", lower case."""
    b = bus_id.strip().lower()
    dom, _, rest = b.rpartition(":")
    dom, _, bus = dom.rpartition(":") if dom.count(":") else ("", "", dom)
    if not dom:
        return b
    return f"{int(dom, 16) & 0xFFFF:04x}:{bus}:{rest}"


def _nvml_gpus() -> Optional[List[Dict]]:
    try:
        import pynvml  # type: ignore
    except ImportError:
        return None
    try:
        pynvml.nvmlInit()
    except Exception:
        return None
    try:
        out = []
        for i in range(pynvml.nvmlDeviceGetCount()):
            h = pynvml.nvmlDeviceGetHandleByIndex(i)
            name = pynvml.nvmlDeviceGetName(h)
            if isinstance(name, bytes):
                name = name.decode()
            mem = pynvml.nvmlDeviceGetMemoryInfo(h)
            g = {"index": i, "name": name, "vendor": "nvidia",
                 "memoryMB": int(mem.total // (1024 * 1024)),
                 "freeMemoryMB": int(mem.free // (1024 * 1024))}
            try:
                bus = pynvml.nvmlDeviceGetPciInfo(h).busId
                g["pci"] = normalize_pci(bus.decode() if isinstance(bus, bytes) else str(bus))
            except Exception:
                pass
            out.append(g)
        return out
    except Exception:
        return None
    finally:
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass


def _torch_gpus() -> List[Dict]:
    if importlib.util.find_spec("torch") is None:
        return []
    try:
        import torch

        if not torch.cuda.is_available():
            return []
        vendor = "amd" if getattr(torch.version, "hip", None) else "nvidia"
        out = []
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            try:
                free, total = torch.cuda.mem_get_info(i)
            except Exception:
                free, total = props.total_memory, props.total_memory
            g = {"index": i, "name": props.name, "vendor": vendor,
                 "memoryMB": int(total // (1024 * 1024)),
                 "freeMemoryMB": int(free // (1024 * 1024))}
            if getattr(props, "pci_bus_id", None) is not None:
                g["pci"] = (f"{int(getattr(props, 'pci_domain_id', 0) or 0):04x}:{int(props.pci_bus_id):02x}:"
                            f"{int(getattr(props, 'pci_device_id', 0) or 0):02x}.0")
            out.append(g)
        return out
    except Exception:
        return []


def list_gpus(indices: Optional[List[int]] = None) -> List[Dict]:
    gpus = _nvml_gpus()
    if gpus is None:
        gpus = _torch_gpus()
    if indices is not None:
        by_index = {g["index"]: g for g in gpus}
        gpus = [by_index[i] for i in indices if i in by_index]
    return gpus


def adapters(sys_root=None) -> List[Dict]:
    """Every graphics card (NVIDIA, AMD, Intel) as ``/v1/info`` "adapters" entries; [] when sysfs can't be read."""
    try:
        from pathlib import Path

        from .gpu_vendors import scan

        return [a.to_info() for a in scan(Path(sys_root) if sys_root else Path("/sys"))]
    except Exception:
        return []


def torch_backend() -> str:
    """"cuda" (NVIDIA), "rocm" (AMD through HIP), "xpu" (Intel) or "cpu" — what the installed PyTorch can use."""
    if importlib.util.find_spec("torch") is None:
        return "cpu"
    try:
        import torch

        if torch.cuda.is_available():
            return "rocm" if getattr(torch.version, "hip", None) else "cuda"
        xpu = getattr(torch, "xpu", None)
        if xpu is not None and xpu.is_available():
            return "xpu"
    except Exception:
        pass
    return "cpu"


_processing_cache: Dict = {}


def processing_device(refresh: bool = False) -> Dict:
    """Where dataset processing (PyTorch models in splatprep) runs here: {"device": "cuda" | "rocm" | "xpu" | "cpu",
    "gpu": name or None, "reason": why not a GPU, or None}; checked once (importing PyTorch takes seconds)."""
    if _processing_cache and not refresh:
        return dict(_processing_cache)
    dev = torch_backend()
    gpu = None
    reason = None
    try:
        from .gpu_vendors import best_non_nvidia, scan
        from pathlib import Path

        ads = scan(Path("/sys"))
        nv = [a for a in ads if a.vendor == "nvidia"]
        other = best_non_nvidia(ads)
        gpu = (nv[0].name if nv else other.name if other else None)
    except Exception:  # noqa: BLE001
        pass
    if dev == "cpu":
        if importlib.util.find_spec("torch") is None:
            reason = "the processing runtime isn't installed"
        elif gpu:
            reason = f"PyTorch can't use the {gpu}"
        else:
            reason = "no GPU found"
    _processing_cache.clear()
    _processing_cache.update(device=dev, gpu=gpu if dev != "cpu" else None, reason=reason, gpuName=gpu)
    return dict(_processing_cache)


def backend_description() -> str:
    try:
        from importlib.metadata import version

        return f"gsplat {version('gsplat')}"
    except Exception:
        return "gsplat (not installed)"


def training_available() -> bool:
    return importlib.util.find_spec("torch") is not None and importlib.util.find_spec("gsplat") is not None
