"""Job execution backends.

The server hands a :class:`JobContext` to a backend's ``run``. The production
:class:`SubprocessBackend` runs each job in a fresh ``spawn``-ed Python
process (one per GPU slot), which keeps the API process free of CUDA state,
returns all GPU memory to the driver after each job, and isolates crashes.
Tests substitute a fake backend.
"""

from __future__ import annotations

import multiprocessing as mp
import queue as queue_mod
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional


class JobCancelled(Exception):
    pass


class JobFailed(RuntimeError):
    pass


class JobPaused(Exception):
    """The trainer saved its full state (``resume.pt`` in the job dir) and stopped; the job can be resumed."""

    def __init__(self, step: int = 0):
        super().__init__(f"paused at step {step}")
        self.step = int(step)


@dataclass
class JobContext:
    job_id: str
    kind: str                      # "train" | "process" | "census"
    job_dir: Path
    dataset_dir: Path              # root of the uploaded dataset
    spec: Dict[str, Any]           # TrainingConfig dict (train) or ProcessingRequest (process)
    device_index: Optional[int]    # GPU index, None if no GPU detected
    status: Callable[..., None]    # update JobStatus fields
    log: Callable[[str], None]
    preview: Callable[[int], None]  # preview.ply for this step has been written
    cancel_event: threading.Event = field(default_factory=threading.Event)
    # train jobs on a pausable backend: set → save state and stop (raise JobPaused); set → write checkpoint.ply
    # of the current splats and call ``checkpoint(step)`` (the backend clears it once forwarded)
    pause_event: threading.Event = field(default_factory=threading.Event)
    checkpoint_event: threading.Event = field(default_factory=threading.Event)
    checkpoint: Callable[[int], None] = lambda step: None
    vulkan_index: Optional[int] = None   # the slot's Vulkan device (splatwork-vk-train --device); None: its own pick
    trainer: Optional[str] = None        # train jobs: "gsplat" | "vulkan" asked for by the job; None: the default
    # processing jobs (gpulease.py): ask for the home GPU before a GPU stage — gpu_request(stage, need_mb, grant)
    # calls grant() once the lease is given; gpu_release() hands it back. None: the job holds a GPU throughout.
    gpu_request: Optional[Callable[[str, int, Callable[[], None]], None]] = None
    gpu_release: Optional[Callable[[], None]] = None


class Backend:
    pausable = False   # train jobs honour JobContext.pause_event / checkpoint_event
    leases = False     # processing jobs use JobContext.gpu_request / gpu_release (else they hold a GPU slot)

    def census_available(self) -> bool:
        return True

    def trainer_choice(self) -> Dict[str, Any]:
        return {"trainer": "gsplat", "reason": "default"}

    def capabilities(self) -> List[str]:
        raise NotImplementedError

    def models(self) -> List[str]:
        return []

    def run(self, ctx: JobContext) -> Dict[str, Any]:
        """Run the job to completion; raise JobCancelled or an exception on failure."""
        raise NotImplementedError


class SubprocessBackend(Backend):
    pausable = True
    leases = True      # processing jobs hold a GPU only around GPU stages (gpulease.py)

    def __init__(self, cancel_grace_seconds: float = 30.0, trainer: str = "auto"):
        self.cancel_grace = cancel_grace_seconds
        self._mp = mp.get_context("spawn")
        self.trainer_setting = trainer
        self._trainer_choice: Optional[Dict[str, Any]] = None
        self._choice_lock = threading.Lock()

    def trainer_choice(self) -> Dict[str, Any]:
        """{"trainer": "gsplat" | "vulkan" | None, "reason"} — chosen once (the Vulkan probe starts the GPU)."""
        with self._choice_lock:
            if self._trainer_choice is None:
                from .vktrain import select

                self._trainer_choice = select(self.trainer_setting)
            return self._trainer_choice

    def census_available(self) -> bool:
        """Auto clean renders with gsplat, so it needs gsplat even when the Vulkan trainer trains."""
        from .gpu import training_available

        return training_available()

    def capabilities(self) -> List[str]:
        from .processing import frames_available, splatprep_available

        caps = []
        trainer = self.trainer_choice()["trainer"]
        if trainer is not None:
            caps.append("train")
        if splatprep_available():
            caps.append("process")
        if frames_available():
            caps.append("frames")
        return caps

    def models(self) -> List[str]:
        from .processing import available_models

        return available_models()

    def run(self, ctx: JobContext) -> Dict[str, Any]:
        q = self._mp.Queue()
        cancel = self._mp.Event()
        pause = self._mp.Event()
        ckpt = self._mp.Event()
        granted = None
        if ctx.kind == "train" and (ctx.trainer or self.trainer_choice()["trainer"]) == "vulkan":
            from .vktrain import subprocess_entry

            args = (str(ctx.dataset_dir), ctx.spec, str(ctx.job_dir), ctx.vulkan_index, q, cancel, pause, ckpt)
        elif ctx.kind == "train":
            from .trainer import subprocess_entry

            device = f"cuda:{ctx.device_index}" if ctx.device_index is not None else "cuda:0"
            args = (str(ctx.dataset_dir), ctx.spec, str(ctx.job_dir), device, q, cancel, pause, ckpt)
        elif ctx.kind == "process":
            from .processing import subprocess_entry

            granted = self._mp.Event() if ctx.gpu_request is not None else None
            args = (str(ctx.dataset_dir), ctx.spec, str(ctx.job_dir), ctx.device_index, q, cancel, granted)
        elif ctx.kind == "census":
            from .census import subprocess_entry

            args = (str(ctx.dataset_dir), ctx.spec, str(ctx.job_dir), ctx.device_index, q, cancel)
        else:
            raise JobFailed(f"unknown job kind {ctx.kind}")

        proc = self._mp.Process(target=subprocess_entry, args=args, daemon=True,
                                name=f"splatworker-{ctx.job_id}")
        proc.start()
        ctx.log(f"started worker process pid {proc.pid}")
        cancel_sent_at: Optional[float] = None
        outcome: Optional[tuple] = None
        try:
            while True:
                if ctx.cancel_event.is_set() and cancel_sent_at is None:
                    cancel.set()
                    cancel_sent_at = time.time()
                    ctx.log("cancellation requested")
                if ctx.pause_event.is_set() and not pause.is_set():
                    pause.set()            # no grace timer: saving the state of a big scene takes a while
                    ctx.log("pause requested")
                if ctx.checkpoint_event.is_set():
                    ctx.checkpoint_event.clear()
                    ckpt.set()
                if cancel_sent_at is not None and time.time() - cancel_sent_at > self.cancel_grace:
                    ctx.log("worker did not stop in time; terminating")
                    proc.terminate()
                    proc.join(10)
                    if proc.is_alive():
                        proc.kill()
                    raise JobCancelled()
                try:
                    kind, payload = q.get(timeout=0.5)
                except queue_mod.Empty:
                    if not proc.is_alive():
                        # drain anything left, then decide
                        try:
                            kind, payload = q.get(timeout=1.0)
                        except queue_mod.Empty:
                            if cancel_sent_at is not None:
                                raise JobCancelled()
                            raise JobFailed(f"worker process exited unexpectedly (code {proc.exitcode})")
                    else:
                        continue
                if kind == "status":
                    ctx.status(**payload)
                elif kind == "log":
                    ctx.log(payload)
                elif kind == "preview":
                    ctx.preview(int(payload))
                elif kind == "checkpoint":
                    ctx.checkpoint(int(payload))
                elif kind == "gpu" and granted is not None:
                    if payload.get("acquire"):
                        ctx.gpu_request(str(payload["acquire"]), int(payload.get("needMB") or 0), granted.set)
                    elif payload.get("release"):
                        ctx.gpu_release()
                elif kind in ("done", "cancelled", "error", "paused"):
                    outcome = (kind, payload)
                    break
        finally:
            proc.join(15)
            if proc.is_alive():
                proc.terminate()
                proc.join(5)
            if ctx.gpu_release is not None:
                ctx.gpu_release()           # whatever happened, the lease ends with the job
        kind, payload = outcome
        if kind == "cancelled":
            raise JobCancelled()
        if kind == "paused":
            raise JobPaused(int(payload or 0))
        if kind == "error":
            raise JobFailed(payload)
        return payload or {}
