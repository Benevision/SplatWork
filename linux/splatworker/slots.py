"""Job slots: which GPU runs which job, one job per GPU at a time.

A slot is one physical GPU. Its key is how ``serve --gpus`` and the app's Preferences → GPUs name it:

* an ``int`` — a CUDA GPU (NVML / PyTorch index). It takes every kind of job (processing and Auto clean need
  CUDA). With the Vulkan trainer chosen, its train jobs run on the matching Vulkan device (``vulkan``: same PCI
  address, else the same name);
* ``"vk:<pci address or name>"`` — a GPU only the Vulkan trainer can use (an AMD or Intel card, an integrated
  GPU). It takes train jobs only, and only while the Vulkan trainer is the one training;
* ``None`` — no GPU found: one slot on the default device (the trainer's own pick).

Slow slots. A much slower GPU next to a faster one (BensPC: a Radeon 610M beside an RTX 4080) must not hold up a
run, e.g. a block run whose blocks the Mac sends to this worker as one device. Speeds are measured, not guessed
(gpuspeed.py: the same timed training on each device): a slot is slow when it measured under SLOW_RATIO of the
fastest slot measured at the same scene size, or only fit a smaller scene; until both are measured, an integrated GPU next to another GPU counts as slow. A slow slot
(``slow``) is never in ``/v1/info`` "gpus" (what the Mac counts as this worker's devices) and takes a train job
only when the job names it (``slot``), or when every fast slot is busy and the job stands alone (not a block —
no ``region`` / ``cameraIds`` — and not a step of a whole-pipeline job). Integrated GPUs are also off by default
("Use all GPUs" means every GPU but those, unless an integrated GPU is the only one); the user ticks them.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Union

SlotKey = Union[int, str, None]
VK_PREFIX = "vk:"
SLOW_RATIO = 0.25       # measured under a quarter of the fastest slot's speed: slow


@dataclass
class Slot:
    key: SlotKey
    name: Optional[str] = None
    cuda: Optional[int] = None        # CUDA index for processing / gsplat / Auto clean
    vulkan: Optional[int] = None      # splatwork-vk-train --device N (None: the program's own pick)
    integrated: bool = False
    slow: bool = False
    memory_mb: Optional[int] = None
    vendor: Optional[str] = None
    device: Optional[Dict[str, Any]] = None   # its --list-devices entry (gpuspeed.device_key)
    speed: Optional[float] = None             # measured steps/s (gpuspeed.py)
    size: Optional[str] = None                # the scene size it was measured at
    relative: Optional[float] = None          # speed / the fastest measured slot's

    @property
    def vulkan_only(self) -> bool:
        return isinstance(self.key, str)

    @property
    def id(self) -> str:
        """The slot's name in ``/v1/info`` "slots" and a train job's ``slot`` field."""
        if self.key is None:
            return "default"
        return self.key if isinstance(self.key, str) else f"cuda:{self.key}"

    def takes_kind(self, kind: str) -> bool:
        return not self.vulkan_only or kind == "train"

    def info(self, busy: bool = False) -> Dict[str, Any]:
        return {"id": self.id, "name": self.name, "kinds": ["train"] if self.vulkan_only else ["train", "process", "census"],
                "cuda": self.cuda, "vulkan": self.vulkan, "integrated": self.integrated, "slow": self.slow,
                "memoryMB": self.memory_mb, "busy": busy,
                # ≥ 1.7.3: measured Vulkan training speed (null: not measured) and its share of the fastest slot
                "stepsPerSecond": self.speed, "relativeSpeed": self.relative, "measuredSize": self.size}


def vulkan_key(dev: Dict[str, Any]) -> str:
    return VK_PREFIX + (dev.get("pci") or str(dev.get("name") or f"device {dev.get('index')}"))


def is_slot_key(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    if isinstance(value, int):
        return value >= 0
    return isinstance(value, str) and value.startswith(VK_PREFIX) and len(value) > len(VK_PREFIX)


def parse_key(text: str) -> SlotKey:
    t = text.strip()
    if t.isdigit():
        return int(t)
    if is_slot_key(t):
        return t
    raise ValueError(f"not a GPU: {text!r} (a CUDA index such as 0, or vk:<PCI address> for a Vulkan GPU)")


def _integrated(dev: Dict[str, Any]) -> bool:
    return bool(dev.get("integrated")) or dev.get("deviceType") == "integrated"


def _label(dev: Dict[str, Any]) -> str:
    return str(dev.get("label") or dev.get("name") or f"Vulkan device {dev.get('index')}")


def _tidy(vk_devices: Optional[Sequence[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    from .vktrain import tidy_devices

    return tidy_devices(list(vk_devices or []))


def _vulkan_usable(dev: Dict[str, Any]) -> bool:
    return dev.get("deviceType") != "cpu" or os.environ.get("SPLATWORK_VK_ALLOW_CPU") == "1"


def match_vulkan(cuda_gpus: Sequence[Dict[str, Any]], vk_devices: Sequence[Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
    """CUDA index → its Vulkan device: the same PCI address (NVML writes the domain with 8 digits, Vulkan with 4:
    both normalised), else the same name, else the same vendor and memory size (±10 %)."""
    from .gpu import normalize_pci

    out: Dict[int, Dict[str, Any]] = {}
    used = set()
    for g in cuda_gpus:
        pci = normalize_pci(str(g["pci"])) if g.get("pci") else None
        for d in vk_devices:
            if d["index"] not in used and pci and d.get("pci") and normalize_pci(str(d["pci"])) == pci:
                out[g["index"]] = d
                used.add(d["index"])
                break
    for g in cuda_gpus:
        if g["index"] in out:
            continue
        for d in vk_devices:
            if d["index"] not in used and str(d.get("name", "")).strip() == str(g.get("name", "")).strip():
                out[g["index"]] = d
                used.add(d["index"])
                break
    for g in cuda_gpus:
        if g["index"] in out or not g.get("memoryMB"):
            continue
        vendor = {"nvidia": 0x10DE, "amd": 0x1002, "intel": 0x8086}.get(g.get("vendor", "nvidia"))
        same = [d for d in vk_devices if d["index"] not in used and int(d.get("vendorID") or 0) == vendor
                and d.get("memoryMB") and abs(int(d["memoryMB"]) - int(g["memoryMB"])) <= 0.1 * int(g["memoryMB"])]
        if len(same) == 1:
            out[g["index"]] = same[0]
            used.add(same[0]["index"])
    return out


def vulkan_only_devices(cuda_gpus: Sequence[Dict[str, Any]], vk_devices: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """The Vulkan devices no CUDA GPU matches (CPU drivers left out unless SPLATWORK_VK_ALLOW_CPU=1)."""
    vk_devices = _tidy(vk_devices)
    matched = {d["index"] for d in match_vulkan(cuda_gpus, vk_devices).values()}
    return [d for d in vk_devices if d["index"] not in matched and _vulkan_usable(d)]


def default_on(dev: Dict[str, Any], others: bool) -> bool:
    """Whether "Use all GPUs" includes this Vulkan-only device: not an integrated GPU, unless it is the only GPU."""
    return not _integrated(dev) or not others


def plan(selected: Optional[Sequence[SlotKey]], cuda_gpus: Sequence[Dict[str, Any]],
         vk_devices: Optional[Sequence[Dict[str, Any]]], trainer: Optional[str],
         speeds: Optional[Dict[str, Dict[str, Any]]] = None) -> List[Slot]:
    """The worker's slots. ``selected``: ``settings.gpus`` (None = every GPU but integrated ones beside others);
    ``speeds``: gpuspeed.load()."""
    vk_devices = _tidy(vk_devices)
    vulkan = trainer == "vulkan"
    match = match_vulkan(cuda_gpus, vk_devices) if vulkan else {}
    chosen = None if selected is None else list(selected)
    slots: List[Slot] = []
    listed = {g["index"] for g in cuda_gpus}
    for k in chosen or []:          # chosen CUDA indices the GPU listing didn't show (as before slots existed)
        if isinstance(k, int) and k not in listed:
            slots.append(Slot(key=k, cuda=k))
    for g in cuda_gpus:
        if chosen is not None and g["index"] not in chosen:
            continue
        d = match.get(g["index"])
        slots.append(Slot(key=g["index"], name=g.get("name"), cuda=g["index"],
                          vulkan=d["index"] if d is not None else None,
                          integrated=bool(d and _integrated(d)),
                          memory_mb=g.get("memoryMB"), vendor=g.get("vendor", "nvidia"), device=d))
    if vulkan:
        extra = vulkan_only_devices(cuda_gpus, vk_devices)
        others = bool(cuda_gpus) or any(not _integrated(d) for d in extra)
        for d in extra:
            key = vulkan_key(d)
            on = default_on(d, others) if chosen is None else key in chosen
            if on:
                slots.append(Slot(key=key, name=_label(d), vulkan=d["index"],
                                  integrated=_integrated(d), memory_mb=d.get("memoryMB"),
                                  vendor=_VENDORS.get(int(d.get("vendorID") or 0), "other"), device=d))
    if not slots:
        return [Slot(key=None)]
    apply_speeds(slots, speeds or {})
    return slots


def apply_speeds(slots: Sequence[Slot], speeds: Dict[str, Dict[str, Any]]) -> None:
    """Set each slot's measured speed, its share of the fastest, and ``slow`` (see the module doc). Speeds compare
    only at the same scene size; a slot that only fit a smaller scene than another slot measured is slow."""
    from .gpuspeed import record_of, size_rank

    recs = {id(s): record_of(s.device, speeds) for s in slots}
    for s in slots:
        r = recs[id(s)]
        s.speed = float(r["stepsPerSecond"]) if r else None
        s.size = r.get("size") if r else None
    measured = [s for s in slots if s.speed]
    top = min((size_rank(s.size) for s in measured), default=None)
    best = max((s.speed for s in measured if size_rank(s.size) == top), default=None)
    for s in slots:
        same = s.speed and size_rank(s.size) == top
        s.relative = round(s.speed / best, 4) if same and best else None
        if len(slots) < 2:
            s.slow = False
        elif s.speed and len(measured) >= 2:
            s.slow = not same or s.relative < SLOW_RATIO
        else:       # not measured against another slot yet: an integrated GPU beside another counts as slow
            s.slow = s.integrated and any(o is not s and not o.integrated for o in slots)


_VENDORS = {0x10DE: "nvidia", 0x1002: "amd", 0x8086: "intel"}


def standalone(spec: Dict[str, Any]) -> bool:
    """A train job a slow slot may take: not a block of a block run, not a step of a whole-pipeline job."""
    cfg = spec.get("config") or {}
    return not spec.get("parentJobId") and not cfg.get("region") and not cfg.get("cameraIds")


def gpu_choices(cuda_gpus: Sequence[Dict[str, Any]], vk_devices: Sequence[Dict[str, Any]],
                selected: Optional[Sequence[SlotKey]],
                speeds: Optional[Dict[str, Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
    """The rows of the app's Preferences → GPUs: {key, title, subtitle, active, integrated}. ``active`` is the
    tick: with "Use all GPUs" (``selected`` None) every GPU but an integrated one beside faster ones."""
    speeds = speeds or {}
    vk_devices = _tidy(vk_devices)
    match = match_vulkan(cuda_gpus, vk_devices)
    rows: List[Dict[str, Any]] = []
    for g in cuda_gpus:
        mem = g.get("memoryMB") or 0
        d = match.get(g["index"])
        rows.append({"key": g["index"], "title": f"GPU {g['index']}: {g.get('name')}",
                     "subtitle": f"{mem / 1024:.0f} GB" if mem else "", "integrated": False,
                     "active": selected is None or g["index"] in selected, "device": d})
    extra = vulkan_only_devices(cuda_gpus, vk_devices)
    others = bool(cuda_gpus) or any(not _integrated(d) for d in extra)
    for d in extra:
        key = vulkan_key(d)
        integrated = _integrated(d)
        if integrated and others:
            sub = ("Integrated GPU, speed not measured yet, so it counts as much slower than the others: it only "
                   "trains a job on its own while they are all busy, never part of a block run. Vulkan trainer only.")
        else:
            sub = "Trains with the Vulkan trainer (training jobs only)."
        rows.append({"key": key, "title": _label(d), "subtitle": sub, "integrated": integrated,
                     "active": default_on(d, others) if selected is None else key in selected, "device": d})
    from .gpuspeed import record_of, size_rank

    for r in rows:
        r["record"] = record_of(r.pop("device"), speeds)
    measured = [r for r in rows if r["record"]]
    top = min((size_rank(r["record"].get("size")) for r in measured), default=None)
    best = max((r["record"]["stepsPerSecond"] for r in measured if size_rank(r["record"].get("size")) == top),
               default=None)
    for r in rows:          # measured speeds replace the guess in the row's note
        rec = r.pop("record")
        if not rec:
            continue
        same = size_rank(rec.get("size")) == top
        share = rec["stepsPerSecond"] / best if same and best else None
        note = f"Measured {rec['stepsPerSecond']:.1f} training steps/s on the {rec.get('size', 'test')} scene"
        note += f" ({share:.0%} of the fastest)." if share is not None and len(measured) > 1 else "."
        if rec.get("peakMemoryMB"):
            note += f" Used {rec['peakMemoryMB'] / 1024:.1f} GB."
        slow = len(measured) > 1 and (share is None or share < SLOW_RATIO)
        if slow:
            note += (" Much slower: it only trains a job on its own while the others are busy, never part of a "
                     "block run.")
        elif r["integrated"]:
            note += " Fast enough to take jobs like the others when ticked. Vulkan trainer only."
        if isinstance(r["key"], int):           # a CUDA GPU: keep its memory size in front
            r["subtitle"] = (r["subtitle"] + " · " if r["subtitle"] else "") + note
        else:
            r["subtitle"] = note
    return rows
