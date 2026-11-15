"""GPU leases for processing jobs: the GPU only around their GPU stages.

A processing job (splatprep: camera solve, masks, frames) runs much of its time on the CPU — COLMAP / GLOMAP mapping
and bundle adjustment, verification, export, video decoding, VHS restoration. It used to hold a GPU slot for the
whole run, so it waited behind training even for its CPU work. Now processing jobs run in a CPU pool (at most
``ServerSettings.cpu_jobs`` at once; COLMAP already uses every core) and take a *lease* on their home GPU only for
stages that need it; training keeps its slot for its whole run, unchanged.

The job process (``processing.subprocess_entry``) sees splatprep's progress events, which name the stage. Before a
stage that needs the GPU (``needs_gpu``) it sends ``("gpu", {"acquire": stage})`` on its queue and blocks until the
worker sets the shared ``granted`` event; when a CPU stage starts it frees PyTorch's cached GPU memory, clears
``granted`` and sends ``("gpu", {"release": True})``. A splatprep event may also say ``"gpu": true | false`` itself,
which wins over the table.

The worker (server.JobManager) grants a lease on the job's home slot when nothing holds that GPU; while a training
job runs there, only when the GPU's free memory covers the stage (``NEED_MB`` + ``MARGIN_MB``), so a short GPU stage
can run beside training. Queued training comes first: a lease waits while a train job could start on that GPU.
"""
from __future__ import annotations

import threading
import time
from typing import Any, Callable, Dict, Optional

# stages that need the GPU, per splatprep task; None = the whole task
GPU_STAGES: Dict[str, Optional[frozenset]] = {
    "sfm": frozenset({"pairs", "features", "matching"}),   # learned retrieval / features / matchers
    "masks": None,                                          # SAM 2, sky, people: all on the GPU
    "maskpreview": None,
    "benchmark": None,
    "frames": frozenset({"splitting 360° video", "extracting stereo frames"}),   # resampling / stereo on torch
}
# rough peak GPU memory of a stage (MB): a lease beside a running training job needs this much free
NEED_MB: Dict[str, int] = {"masks": 6000, "maskpreview": 4000, "features": 3000, "matching": 4000, "pairs": 2000}
DEFAULT_NEED_MB = 4000
MARGIN_MB = 1024


def needs_gpu(task: str, stage: str, event: Optional[Dict[str, Any]] = None) -> bool:
    if event is not None and isinstance(event.get("gpu"), bool):
        return event["gpu"]
    if task not in GPU_STAGES:
        return True                       # unknown task: as before, the GPU for the whole run
    stages = GPU_STAGES[task]
    return stages is None or stage in stages


def need_mb(task: str, stage: str) -> int:
    return NEED_MB.get(stage, NEED_MB.get(task, DEFAULT_NEED_MB))


class Gate:
    """Job-process side: call ``stage(task, stage, event)`` for every progress event; blocks while waiting for the
    GPU. ``status(**fields)`` reports "waiting for GPU" / which resource the job is on."""

    def __init__(self, queue, granted, cancelled: Callable[[], bool], status: Callable[..., None]):
        self.queue = queue
        self.granted = granted            # multiprocessing.Event shared with the worker
        self.cancelled = cancelled
        self.status = status
        self.holding = False
        self.resource: Optional[str] = None

    def stage(self, task: str, stage: str, event: Optional[Dict[str, Any]] = None) -> None:
        want = needs_gpu(task, stage, event)
        if want and not self.holding:
            self._set("waiting-gpu", f"{stage}: waiting for the GPU")
            self.queue.put(("gpu", {"acquire": stage, "needMB": need_mb(task, stage)}))
            while not self.granted.wait(0.25):
                if self.cancelled():
                    raise KeyboardInterrupt("cancelled")
            self.holding = True
            self._set("gpu", None)
        elif not want and self.holding:
            self.release()
            self._set("cpu", None)
        elif self.resource is None:
            self._set("gpu" if want else "cpu", None)

    def release(self) -> None:
        if not self.holding:
            return
        free_torch_cache()
        self.holding = False
        self.granted.clear()
        self.queue.put(("gpu", {"release": True}))

    def _set(self, resource: str, message: Optional[str]) -> None:
        self.resource = resource
        fields: Dict[str, Any] = {"resource": resource}
        if message:
            fields["message"] = message
        self.status(**fields)


def free_torch_cache() -> None:
    import sys

    torch = sys.modules.get("torch")
    if torch is None:
        return
    try:
        if torch.cuda.is_available() and torch.cuda.is_initialized():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001
        pass


class Pending:
    """Worker side: one lease request waiting for its GPU."""

    def __init__(self, job_id: str, slot, stage: str, need: int, grant: Callable[[], None]):
        self.job_id = job_id
        self.slot = slot
        self.stage = stage
        self.need = need
        self.grant = grant
        self.since = time.monotonic()
        self.lock = threading.Lock()
