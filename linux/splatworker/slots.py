"""Job slots: which GPU runs which job, one job per GPU at a time.

A slot is one physical GPU. Its key is how ``serve --gpus`` and the app's Preferences → GPUs name it:

* an ``int`` — a CUDA GPU (NVML / PyTorch index). It takes every kind of job (processing and Auto clean need
  CUDA). With the Vulkan trainer chosen, its train jobs run on the matching Vulkan device (``vulkan``: same PCI
  address, else the same name);
* ``"vk:<pci address or name>"`` — a GPU only the Vulkan trainer can use (an AMD or Intel card, an integrated
  GPU). It takes train jobs only, and only while the Vulkan trainer is the one training;
* ``None`` — no GPU found: one slot on the default device (the trainer's own pick).

Slower slots. A much slower GPU next to a faster one (BensPC: a Radeon 610M beside an RTX 4080) must not hold up a
run, e.g. a block run whose blocks the Mac sends to this worker as one device. Since 1.7.8 a slower slot (user choice
"auto") starts a waiting train job — block jobs too — only while every faster slot is busy and the job fits its
memory, and the job moves to a faster slot when one frees up (server.JobManager, Vulkan trainer: pause, resume on
the other GPU); with the user's "always" it is a slot like any other. The rest of this paragraph is the 1.7.3
rule for the record. Speeds are measured, not guessed
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
    slower: bool = False                      # much slower than another slot (measured, or integrated unmeasured)
    user_choice: str = "auto"                 # "auto": start waiting jobs while faster GPUs are busy; "always"
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
                "stepsPerSecond": self.speed, "relativeSpeed": self.relative, "measuredSize": self.size,
                # ≥ 1.7.8: "slower" than another slot; the user's choice for it ("auto" | "always": then not slow)
                "slower": self.slower, "userChoice": self.user_choice}


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
         speeds: Optional[Dict[str, Dict[str, Any]]] = None,
         modes: Optional[Dict[str, str]] = None) -> List[Slot]:
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
    for s in slots:
        s.user_choice = (modes or {}).get(str(s.key), "auto")
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
            s.slower = False
        elif s.speed and len(measured) >= 2:
            s.slower = bool(not same or s.relative < SLOW_RATIO)
        else:       # not measured against another slot yet: an integrated GPU beside another counts as slower
            s.slower = s.integrated and any(o is not s and not o.integrated for o in slots)
        s.slow = s.slower and s.user_choice != "always"


_VENDORS = {0x10DE: "nvidia", 0x1002: "amd", 0x8086: "intel"}


def estimate_mb(cfg: Dict[str, Any], photos: int, width: int, height: int) -> int:
    """Rough peak memory (MB) of a Vulkan training job: splats (parameters, two Adam moments, gradients,
    densification and sorting buffers ~1.1 KB each at maxGaussians), the photos at their training size (8-bit
    RGBA) and the render buffers."""
    from .config import TrainingConfig

    try:
        tc = TrainingConfig.from_dict({k: v for k, v in cfg.items() if k in ("maxGaussians", "maxImageSize")})
        splats, max_side = tc.maxGaussians, tc.maxImageSize
    except Exception:  # noqa: BLE001
        splats, max_side = int(cfg.get("maxGaussians") or 3_000_000), int(cfg.get("maxImageSize") or 1600)
    scale = min(1.0, max_side / max(width, height, 1)) if width and height else 1.0
    px = int(width * scale) * int(height * scale) if width and height else 1600 * 1200
    return int((splats * 1100 + photos * px * 4 + px * 64) / 2**20) + 512


def budget_mb(slot: "Slot", system_mb: Optional[int] = None) -> Optional[int]:
    """Memory a job may use on ``slot``: the device's; an integrated GPU shares the system RAM, so at most a
    quarter of it (or its own carve-out, if larger)."""
    mem = slot.memory_mb
    if slot.integrated:
        if system_mb is None:
            try:
                system_mb = int(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**20)
            except (ValueError, OSError, AttributeError):
                system_mb = None
        if system_mb:
            return max(int(mem or 0), system_mb // 4)
    return int(mem) if mem else None


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
                     "active": selected is None or g["index"] in selected, "device": d, "slower": False})
    extra = vulkan_only_devices(cuda_gpus, vk_devices)
    others = bool(cuda_gpus) or any(not _integrated(d) for d in extra)
    for d in extra:
        key = vulkan_key(d)
        integrated = _integrated(d)
        if integrated and others:
            sub = ("Integrated GPU, speed not measured yet, so it counts as much slower than the others. Vulkan "
                   "trainer only.")
        else:
            sub = "Trains with the Vulkan trainer (training jobs only)."
        rows.append({"key": key, "title": _label(d), "subtitle": sub, "integrated": integrated,
                     "active": default_on(d, others) if selected is None else key in selected, "device": d,
                     "slower": integrated and others})
    from .gpuspeed import record_of, size_rank

    for r in rows:
        r["record"] = record_of(r.pop("device"), speeds)
    measured = [r for r in rows if r["record"]]
    top = min((size_rank(r["record"].get("size")) for r in measured), default=None)
    best = max((r["record"]["stepsPerSecond"] for r in measured if size_rank(r["record"].get("size")) == top),
               default=None)
    fastest = next((r["title"].split(": ", 1)[-1] for r in measured if size_rank(r["record"].get("size")) == top
                    and r["record"]["stepsPerSecond"] == best), None)
    for r in rows:          # measured speeds replace the guess in the row's note
        rec = r.pop("record")
        if not rec:
            continue
        same = size_rank(rec.get("size")) == top
        share = rec["stepsPerSecond"] / best if same and best else None
        note = f"Measured {rec['stepsPerSecond']:.1f} training steps/s on the {rec.get('size', 'test')} scene"
        if share is not None and len(measured) > 1:
            note += f" ({share:.0%} of the {fastest})." if fastest and share < 1 else f" ({share:.0%} of the fastest)."
        else:
            note += "."
        if rec.get("peakMemoryMB"):
            note += f" Used {rec['peakMemoryMB'] / 1024:.1f} GB."
        slow = len(measured) > 1 and (share is None or share < SLOW_RATIO)
        r["slower"] = slow if len(measured) > 1 else r["slower"]
        if slow and share is not None and fastest:
            r["share"] = f"about {share:.0%} of the {fastest}"
        if not slow and r["integrated"]:
            note += " Fast enough to take jobs like the others when ticked. Vulkan trainer only."
        if isinstance(r["key"], int):           # a CUDA GPU: keep its memory size in front
            r["subtitle"] = (r["subtitle"] + " · " if r["subtitle"] else "") + note
        else:
            r["subtitle"] = note
    for r in rows:
        if r["slower"]:
            r["subtitle"] += (" Slower GPU: starts waiting jobs while the faster ones are busy. \"Always use\": "
                              "like any other GPU (a block run then waits for its slowest block).")
    return rows
