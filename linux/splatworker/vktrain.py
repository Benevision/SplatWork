"""The Vulkan trainer (``splatwork-vk-train``, built from ``linux/vktrain``) as a training backend.

Which trainer trains (``serve --trainer``, the app's Preferences → Trainer):

* ``gsplat`` — the CUDA trainer (``trainer.py``: PyTorch + gsplat, from the NVIDIA runtime);
* ``vulkan`` — the Vulkan trainer, a native program that needs no PyTorch: any GPU with a Vulkan 1.2 driver
  (AMD through RADV, Intel through ANV, NVIDIA too). Experimental: it has been checked against gsplat on Mesa's
  software driver only, not yet on real GPUs (docs/dev/GPU_SUPPORT.md);
* ``auto`` (default) — gsplat when the computer has an NVIDIA card, otherwise the Vulkan trainer when it finds
  a GPU (not a CPU-only Vulkan driver), otherwise gsplat if it happens to be installed (the experimental ROCm
  training runtime). NVIDIA computers keep gsplat; Vulkan on NVIDIA is opt-in.

A train job on the Vulkan trainer runs like a CUDA one: one process per job (``subprocess_entry``, spawned by
``backends.SubprocessBackend``) that starts the program, turns its JSON lines into the same queue messages
``trainer.subprocess_entry`` sends, and passes pause / checkpoint / cancel on as stdin commands. The job dir gets
the same files (``preview.splat``, ``checkpoint.ply``, ``result.ply`` + skybox sidecars), except that a paused
job's state is ``resume.vkstate`` instead of ``resume.pt``.

Depth priors: the program can't run Depth Anything; when PyTorch and ``transformers`` are installed (the
processing runtime) this module fills the dataset's ``depth-prior/`` cache first, as ``trainer.py`` does, and the
program reads it.
"""
from __future__ import annotations

import json
import os
import queue as queue_mod
import shutil
import subprocess
import threading
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

BINARY_NAME = "splatwork-vk-train"
INTERFACE = 1                      # VKTRAIN_INTERFACE in linux/vktrain/CMakeLists.txt
TRAINERS = ("auto", "gsplat", "vulkan")
BINARY_ENV = "SPLATWORK_VK_TRAIN"  # path of the program (tests, a development build)
ALLOW_CPU_ENV = "SPLATWORK_VK_ALLOW_CPU"   # "1": a CPU Vulkan driver (lavapipe) counts as a device (tests)
DEVICE_ENV = "SPLATWORK_VK_DEVICE"  # Vulkan device index for --device (default: the program's pick)

EXIT_CODES = {0: "done", 1: "error", 2: "config", 3: "paused", 4: "cancelled"}


def binary_path() -> Optional[str]:
    env = os.environ.get(BINARY_ENV)
    if env:
        return env if os.access(env, os.X_OK) else None
    found = shutil.which(BINARY_NAME)
    if found:
        return found
    app = Path("/app/bin") / BINARY_NAME           # Flatpak, in case PATH was changed
    return str(app) if os.access(app, os.X_OK) else None


_info_cache: Dict[str, Dict[str, Any]] = {}
_info_lock = threading.Lock()


def device_info(binary: Optional[str] = None, timeout: float = 60.0, refresh: bool = False) -> Dict[str, Any]:
    """``splatwork-vk-train --info`` as a dict with ``ok``; cached per program path (the probe starts Vulkan)."""
    binary = binary or binary_path()
    if not binary:
        return {"ok": False, "error": f"{BINARY_NAME} is not installed"}
    with _info_lock:
        if not refresh and binary in _info_cache:
            return _info_cache[binary]
        try:
            r = subprocess.run([binary, "--info"], capture_output=True, text=True, timeout=timeout)
            line = next((ln for ln in reversed(r.stdout.splitlines()) if ln.startswith("{")), "")
            if r.returncode != 0 or not line:
                err = (r.stderr or r.stdout).strip().splitlines()
                info = {"ok": False, "error": (err[-1] if err else f"exit code {r.returncode}")[:300]}
            else:
                info = {**json.loads(line), "ok": True}
        except (OSError, subprocess.SubprocessError, ValueError) as e:
            info = {"ok": False, "error": f"{type(e).__name__}: {e}"[:300]}
        _info_cache[binary] = info
        return info


_list_cache: Dict[str, Dict[str, Any]] = {}


def list_devices(binary: Optional[str] = None, timeout: float = 60.0, refresh: bool = False) -> Dict[str, Any]:
    """``splatwork-vk-train --list-devices``: {"ok", "devices": [{index, name, deviceType, memoryMB, vendorID,
    deviceID, driver, pci}], "error"?}; cached per program path. CPU drivers are listed (slots.py leaves them out)."""
    binary = binary or binary_path()
    if not binary:
        return {"ok": False, "devices": [], "error": f"{BINARY_NAME} is not installed"}
    with _info_lock:
        if not refresh and binary in _list_cache:
            return _list_cache[binary]
        try:
            r = subprocess.run([binary, "--list-devices"], capture_output=True, text=True, timeout=timeout)
            line = next((ln for ln in reversed(r.stdout.splitlines()) if ln.startswith("{")), "")
            if r.returncode != 0 or not line:
                err = (r.stderr or r.stdout).strip().splitlines()
                out = {"ok": False, "devices": [], "error": (err[-1] if err else f"exit code {r.returncode}")[:300]}
            else:
                data = json.loads(line)
                devs = [d for d in data.get("devices") or [] if isinstance(d, dict) and isinstance(d.get("index"), int)]
                try:
                    from .gpu import adapters

                    found = adapters()
                except Exception:  # noqa: BLE001
                    found = []
                out = {"ok": True, "devices": tidy_devices(devs, found)}
        except (OSError, subprocess.SubprocessError, ValueError) as e:
            out = {"ok": False, "devices": [], "error": f"{type(e).__name__}: {e}"[:300]}
        _list_cache[binary] = out
        return out


def tidy_devices(devs: List[Dict[str, Any]], adapters: Optional[List[Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
    """``--list-devices`` entries, one per physical GPU, with ``integrated`` and a readable ``label``.

    The Flatpak's Vulkan loader can see a GPU twice (two driver copies — BensPC listed the RTX 4080 and the Ryzen's
    Radeon 610M twice each): entries with the same PCI address (else the same device UUID) are one GPU; the one
    with the newest Vulkan / driver version is kept (ties: the lower index), the others' indices go in
    ``duplicates``. The system's PCI scan (``gpu.adapters()``) marks integrated GPUs (RADV calls the 610M after the
    CPU, "AMD Ryzen 9 9950X3D 16-Core Processor (RADV RAPHAEL_MENDOCINO)") and names them."""
    from .gpu import normalize_pci

    by_pci = {normalize_pci(a["pci"]): a for a in adapters or [] if a.get("pci")}
    groups: Dict[Any, List[Dict[str, Any]]] = {}
    order: List[Any] = []
    for d in devs:
        d = dict(d)
        if d.get("pci"):
            d["pci"] = normalize_pci(str(d["pci"]))
        key = ("pci", d["pci"]) if d.get("pci") else ("uuid", d["uuid"]) if d.get("uuid") else ("index", d["index"])
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(d)
    out = []
    for key in order:
        group = groups[key]
        best = max(group, key=lambda e: (int(e.get("apiVersion") or 0), int(e.get("driverVersion") or 0),
                                         -int(e["index"])))
        best = dict(best)
        others = sorted(e["index"] for e in group if e is not best and e["index"] != best["index"])
        if others:
            best["duplicates"] = others
        a = by_pci.get(best.get("pci") or "")
        best["integrated"] = (bool(best.get("integrated")) or best.get("deviceType") == "integrated"
                              or bool(a and a.get("integrated")))
        label = str(best.get("label") or best.get("name") or f"Vulkan device {best['index']}")
        if best["integrated"] and a and a.get("name") and a["name"] not in label:
            label = f"{a['name']} (integrated)"
        best["label"] = label
        out.append(best)
    return out


def usable(info: Dict[str, Any]) -> Optional[str]:
    """Why the probed device can't train (None = it can)."""
    if not info.get("ok"):
        return info.get("error") or "the Vulkan trainer did not start"
    if info.get("interface") != INTERFACE:
        return f"{BINARY_NAME} speaks interface {info.get('interface')}, this worker needs {INTERFACE}"
    if info.get("deviceType") == "cpu" and os.environ.get(ALLOW_CPU_ENV) != "1":
        return f"only a CPU Vulkan driver was found ({info.get('device')}), no GPU"
    return None


def available() -> bool:
    return usable(device_info()) is None


def _nvidia_present() -> bool:
    from .gpu import adapters, list_gpus

    if any(a.get("vendor") == "nvidia" for a in adapters()):
        return True
    try:
        return any(g.get("vendor", "nvidia") == "nvidia" for g in list_gpus())
    except Exception:  # noqa: BLE001
        return False


def select(setting: str = "auto") -> Dict[str, Any]:
    """The trainer for ``setting``: {"trainer": "gsplat" | "vulkan" | None, "reason": str}."""
    from .gpu import training_available

    setting = setting if setting in TRAINERS else "auto"
    gsplat_ok = training_available()
    if setting == "gsplat":
        return {"trainer": "gsplat" if gsplat_ok else None,
                "reason": "chosen in settings" if gsplat_ok else "gsplat isn't installed (set up the GPU runtime)"}
    if setting == "vulkan":
        why = usable(device_info())
        return {"trainer": None if why else "vulkan", "reason": why or "chosen in settings"}
    if _nvidia_present():
        return {"trainer": "gsplat" if gsplat_ok else None,
                "reason": "NVIDIA GPU" if gsplat_ok else "NVIDIA GPU; the CUDA runtime isn't installed"}
    why = usable(device_info())
    if why is None:
        return {"trainer": "vulkan", "reason": "no NVIDIA GPU: the Vulkan trainer"}
    if gsplat_ok:
        return {"trainer": "gsplat", "reason": f"Vulkan trainer unavailable ({why})"}
    return {"trainer": None, "reason": why}


def describe(choice: Dict[str, Any]) -> Dict[str, Any]:
    """``/v1/info`` "trainer": {kind, experimental, reason, device?, driver?, memoryMB?, vendorID?}."""
    out: Dict[str, Any] = {"kind": choice.get("trainer"), "reason": choice.get("reason"),
                           "experimental": choice.get("trainer") == "vulkan"}
    if choice.get("trainer") == "vulkan":
        info = device_info()
        for k in ("device", "driver", "deviceType", "memoryMB", "vendorID", "subgroupSize", "nativeAtomics"):
            if k in info:
                out[k] = info[k]
    return out


VENDOR_IDS = {0x1002: "amd", 0x8086: "intel", 0x10DE: "nvidia"}


def gpu_entry() -> Optional[Dict[str, Any]]:
    """The Vulkan device as a ``/v1/info`` "gpus" entry (when PyTorch lists no GPU, e.g. AMD without ROCm)."""
    info = device_info()
    if usable(info) is not None:
        return None
    mem = int(info.get("memoryMB") or 0)
    free = mem
    try:
        from .gpu import adapters

        for a in adapters():
            if a.get("vendor") == VENDOR_IDS.get(int(info.get("vendorID") or 0)) and a.get("freeMemoryMB"):
                free = int(a["freeMemoryMB"])
                mem = int(a.get("memoryMB") or mem)
                break
    except Exception:  # noqa: BLE001
        pass
    return {"name": info.get("device", "Vulkan GPU"), "memoryMB": mem, "freeMemoryMB": free,
            "vendor": VENDOR_IDS.get(int(info.get("vendorID") or 0), "other")}


# ---------------------------------------------------------------------------
# depth priors (Depth Anything through PyTorch, when the processing runtime has it)
# ---------------------------------------------------------------------------

def fill_depth_priors(dataset_dir: str, config: Dict[str, Any], log, status, should_stop) -> None:
    from .config import TrainingConfig
    from .dataset import parse_dataset, select_cameras

    cfg = TrainingConfig.from_dict(config)
    if cfg.normalWeight <= 0 or cfg.depthPriors not in ("auto", "on"):
        return
    ds = parse_dataset(dataset_dir)
    region_box = cfg.region.expanded() if cfg.region is not None else None
    cams = select_cameras(ds, cfg.cameraIds, region_box)
    if any(c.depth for c in cams if not c.holdout):
        return                               # sensor depth: no priors (as trainer.py)
    from . import depthprior

    if not depthprior.available():
        if cfg.depthPriors == "on":
            log("depth priors: the depth model isn't installed on this worker (install the processing runtime)")
        return
    from .gpu import torch_backend

    device = {"cuda": "cuda:0", "rocm": "cuda:0", "xpu": "xpu:0"}.get(torch_backend(), "cpu")
    pts = None
    if ds.points:
        try:
            from .ply import read_point_cloud

            pts, _ = read_point_cloud(ds.resolve(ds.points))
        except Exception as e:  # noqa: BLE001
            log(f"depth priors: could not read the dataset's points ({e})")
    priors = depthprior.DepthPriors(ds.root / "depth-prior", log)
    try:
        got = priors.compute(ds, cams, pts, device=device, should_stop=should_stop, status=status)
        log(f"depth priors: {got} of {len(cams)} photos have a predicted depth map ({device})")
    except Exception as e:  # noqa: BLE001 — a missing model download etc. must not fail the training
        log(f"depth priors unavailable ({type(e).__name__}: {e}); training without them")


# ---------------------------------------------------------------------------
# subprocess entry (the job process; same queue protocol as trainer.subprocess_entry)
# ---------------------------------------------------------------------------

def command(binary: str, dataset_dir: str, job: Path, config_file: Path, device: Optional[int] = None) -> List[str]:
    """``device``: the job slot's Vulkan device index; without one, SPLATWORK_VK_DEVICE, else the program's pick."""
    args = [binary, "train", "--dataset", str(dataset_dir), "--out", str(job / "result.ply"),
            "--config", str(config_file), "--job-dir", str(job)]
    if (job / "resume.vkstate").is_file():
        args += ["--resume", str(job / "resume.vkstate")]
    if device is not None:
        args += ["--device", str(int(device))]
    elif os.environ.get(DEVICE_ENV, "").strip().isdigit():
        args += ["--device", os.environ[DEVICE_ENV].strip()]
    return args


def subprocess_entry(dataset_dir: str, config: dict, job_dir: str, device_index: Optional[int],
                     queue, cancel_event, pause_event=None, checkpoint_event=None) -> None:
    job = Path(job_dir)
    try:
        binary = binary_path()
        if not binary:
            queue.put(("error", f"the Vulkan trainer ({BINARY_NAME}) is not installed"))
            return
        if not (job / "resume.vkstate").is_file():
            try:
                fill_depth_priors(dataset_dir, config, log=lambda m: queue.put(("log", m)),
                                  status=lambda m: queue.put(("status", {"message": m})),
                                  should_stop=cancel_event.is_set)
            except Exception as e:  # noqa: BLE001
                queue.put(("log", f"depth priors skipped ({type(e).__name__}: {e})"))
        if cancel_event.is_set():
            queue.put(("cancelled", None))
            return
        cfg_file = job / "vk-config.json"
        cfg_file.write_text(json.dumps(config), encoding="utf-8")
        run_program(command(binary, dataset_dir, job, cfg_file, device_index), queue, cancel_event, pause_event,
                    checkpoint_event)
    except BaseException as e:  # noqa: BLE001
        queue.put(("log", traceback.format_exc()))
        queue.put(("error", str(e) or e.__class__.__name__))


def run_program(argv: List[str], queue, cancel_event, pause_event=None, checkpoint_event=None,
                poll: float = 0.2) -> None:
    """Runs the program and forwards its JSON lines; puts exactly one outcome message on ``queue``."""
    queue.put(("log", "Vulkan trainer (experimental): " + " ".join(argv[1:])))
    proc = subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            text=True, bufsize=1)
    lines: "queue_mod.Queue[Optional[str]]" = queue_mod.Queue()
    stderr_tail: List[str] = []

    def pump_stdout():
        for ln in proc.stdout:
            lines.put(ln)
        lines.put(None)

    def pump_stderr():
        for ln in proc.stderr:
            ln = ln.rstrip()
            if ln:
                stderr_tail.append(ln)
                del stderr_tail[:-20]
                queue.put(("log", ln))

    threading.Thread(target=pump_stdout, daemon=True).start()
    threading.Thread(target=pump_stderr, daemon=True).start()

    sent = set()

    def send(cmd: str) -> None:
        try:
            proc.stdin.write(cmd + "\n")
            proc.stdin.flush()
        except (OSError, ValueError):
            pass

    outcome = None
    eof = False
    while not eof:
        if cancel_event.is_set() and "cancel" not in sent:
            sent.add("cancel")
            send("cancel")
        if pause_event is not None and pause_event.is_set() and "pause" not in sent:
            sent.add("pause")
            send("pause")
        if checkpoint_event is not None and checkpoint_event.is_set():
            checkpoint_event.clear()
            send("checkpoint")
        try:
            ln = lines.get(timeout=poll)
        except queue_mod.Empty:
            continue
        if ln is None:
            eof = True
            break
        ln = ln.strip()
        if not ln.startswith("{"):
            if ln:
                queue.put(("log", ln))
            continue
        try:
            msg = json.loads(ln)
        except ValueError:
            queue.put(("log", ln))
            continue
        kind = msg.pop("type", None)
        if kind == "status":
            queue.put(("status", msg))
        elif kind == "log":
            queue.put(("log", str(msg.get("message", ""))))
        elif kind in ("preview", "checkpoint"):
            queue.put((kind, int(msg.get("step") or 0)))
        elif kind == "done":
            msg["trainer"] = "vulkan"
            outcome = ("done", msg)
        elif kind == "paused":
            outcome = ("paused", int(msg.get("step") or 0))
        elif kind == "cancelled":
            outcome = ("cancelled", None)
        elif kind == "error":
            outcome = ("error", str(msg.get("message") or "Vulkan trainer error"))
    try:
        rc = proc.wait(timeout=60)
    except subprocess.TimeoutExpired:
        proc.kill()
        rc = proc.wait()
    time.sleep(0.05)                       # let the stderr pump finish its last lines
    if outcome is None:
        kind = EXIT_CODES.get(rc, "error")
        if kind == "cancelled" or (rc < 0 and cancel_event.is_set()):
            outcome = ("cancelled", None)
        elif kind == "paused":
            outcome = ("paused", 0)
        else:
            tail = stderr_tail[-1] if stderr_tail else ""
            outcome = ("error", f"the Vulkan trainer stopped (exit code {rc}){': ' + tail if tail else ''}")
    queue.put(outcome)


def train_cli(dataset_dir: str, out: str, config: Dict[str, Any], device: Optional[str] = None) -> int:
    """``splatworker train --trainer vulkan``: the program with this worker's config, output straight through."""
    binary = binary_path()
    if not binary:
        print(f"{BINARY_NAME} is not installed", flush=True)
        return 1
    import tempfile

    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
        json.dump(config, fh)
        cfg_file = fh.name
    try:
        fill_depth_priors(dataset_dir, config, log=lambda m: print(m, flush=True),
                          status=lambda m: None, should_stop=lambda: False)
    except Exception as e:  # noqa: BLE001
        print(f"depth priors skipped ({type(e).__name__}: {e})", flush=True)
    argv = [binary, "train", "--dataset", str(dataset_dir), "--out", str(out), "--config", cfg_file]
    if device and device.isdigit():
        argv += ["--device", device]
    try:
        return subprocess.call(argv)
    finally:
        os.unlink(cfg_file)
