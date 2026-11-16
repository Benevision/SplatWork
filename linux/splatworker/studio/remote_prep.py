"""Preparing a project's dataset on another worker ("Process on" a worker, as on the Mac: ``DatasetBuilder`` with
``PrepOptions.compute = .remote``).

Video splitting (and home video), learned masks, the sky and the camera solve run on the chosen worker; photo
import, 360° photo splitting, stereo, fixed-region masks and the dataset's last steps stay on this PC. Only what
changed goes back and forth:

* the project's ``images/`` go up once and stay on the worker — remembered in the same uploads record
  (``compute.UPLOADS_FILE``) training on that worker uses, so its training later sends no photos either; a video
  goes up as it is (``POST /v1/datasets/file``), and its extracted frames become that stored copy;
* each step sends only the extra folders it reads (stacked masks, the sky layer, priors), composed on the worker
  with the stored images (``POST /v1/datasets/compose``), and brings back only its output (``output.zip``, without
  the images where it doesn't change them).

Steps the worker can't do run here: a model it doesn't have, a video it can't split, or any error while splitting
or masking. A camera solve that fails on the worker fails the run (as on the Mac).

When the same worker also trains (*Train on* = *Process on*) and it has the ``pipeline`` capability (≥ 1.6.0), the
whole run is one job there (``POST /v1/pipeline``, :func:`pipeline_ineligible` / :meth:`RemotePrep.whole_pipeline`):
the photos (ingested here) or the video go up once with every setting, the worker runs frames → masks → sky →
camera solve → training by itself, and this PC fetches the prepared dataset and then the result — riding out the
worker being unreachable, and re-attaching to the job when the project is prepared again.
"""
from __future__ import annotations

import hashlib
import json
import time
import uuid
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any, Dict, List, Optional, Sequence
from urllib.parse import quote

from .client import ApiError
from .compute import UPLOADS_FILE, _signature, write_zip

FIND_SECONDS = 20.0       # waiting for the chosen worker to show up on the network (the Mac waits 20 s)


PIPELINE_SOURCES = ("photos", "photos360", "video", "video360")
PIPELINE_OFFLINE = 24 * 3600.0       # the worker may be unreachable this long while the job runs there
STAGE_TITLES = {"masks": None, "sky": "Finding the sky", "sfm": "Solving cameras", "static": "Fixed parts of the frame",
                "level": "Finalising dataset", "train": "Training"}


def pipeline_ineligible(o, device, train: bool) -> Optional[str]:
    """Why this run can't be one whole-pipeline job on ``device`` (None = it can). As the Mac's
    ``RemotePipeline.ineligibility``: training must go to the same worker, the worker needs the ``pipeline``
    capability, and every stage must be one it runs as this PC would."""
    if not train or (getattr(o, "train_on", "local") or "local") != o.process_on:
        return "training runs elsewhere"
    if "pipeline" not in device.capabilities:
        return f"{device.name} needs SplatWork for Linux 1.6 or newer for that"
    if o.source_kind not in PIPELINE_SOURCES:
        return "this kind of source is prepared step by step"
    models = [m for m in o.mask_models if m and m != "none"]
    if len(models) > 1:
        return "it runs one mask model and this project stacks several"
    have = worker_models(device.info)
    need = models + [o.features] + (["sky"] if o.sky else [])
    lacking = [m for m in need if have is not None and m not in have]
    if lacking:
        return f"{device.name} doesn't have {' or '.join(lacking)}"
    return None


def options_fingerprint(o) -> str:
    d = o.to_dict() if hasattr(o, "to_dict") else dict(vars(o))
    return hashlib.sha1(json.dumps(d, sort_keys=True, default=str).encode()).hexdigest()[:16]


class StepUnavailable(RuntimeError):
    """This step can't run on the worker (the pipeline then runs it on this PC)."""


def worker_models(info: Optional[Dict[str, Any]]) -> Optional[List[str]]:
    m = (info or {}).get("models")
    return list(m) if isinstance(m, list) else None


def missing_models(info: Optional[Dict[str, Any]], mask_models: Sequence[str], sky: bool,
                   features: Optional[str]) -> List[str]:
    """The models a run would need that the worker hasn't got (its /v1/info "models"); unknown list → none."""
    have = worker_models(info)
    if have is None:
        return []
    need = [m for m in mask_models if m and m != "none"] + (["sky"] if sky else []) + ([features] if features else [])
    return [m for m in dict.fromkeys(need) if m not in have]


def process_on_choices(reg):
    """("local", "This PC") and every known worker: "name", "name (can't process)", "name (offline)"."""
    choices = [("local", "This PC")]
    for d in reg.devices() if reg is not None else []:
        if d.kind != "linux" and not d.online:
            continue
        if d.online:
            label = d.name if "process" in d.capabilities else f"{d.name} (can't process)"
        else:
            label = f"{d.name} (offline)"
        choices.append((d.id, label))
    return choices


def find_device(registry, device_id: str, wait: float = FIND_SECONDS, sleep=time.sleep):
    """The chosen worker once it answers (checked again for up to ``wait`` s), else None."""
    end = time.monotonic() + wait
    while True:
        d = registry.get(device_id) if registry is not None else None
        if d is not None and d.online:
            return d
        if time.monotonic() >= end:
            return None
        if d is not None:
            try:
                registry.check(d.id)
            except Exception:  # noqa: BLE001
                pass
        sleep(0.5)


class RemotePrep:
    def __init__(self, pipeline, device):
        self.p = pipeline
        self.d = device
        self.name = device.name
        self.client = device.client(timeout=120)
        self.ds: Path = pipeline.project.dataset_dir
        self._cleanup: List[str] = []

    # ------------------------------------------------------------ helpers
    @property
    def caps(self) -> List[str]:
        return self.d.capabilities

    def has_model(self, model: Optional[str]) -> bool:
        have = worker_models(self.d.info)
        return have is None or not model or model in have

    def _progress(self, what: str):
        last = {"t": 0.0}

        def cb(sent: int, total: int) -> None:
            now = time.monotonic()
            if now - last["t"] > 0.5 or sent == total:
                last["t"] = now
                self.p._progress(sent / max(total, 1), f"Uploading {what} to {self.name}: {sent / 2**20:.0f} / "
                                                       f"{total / 2**20:.0f} MB")
            self.p._check()
        return cb

    def _tmp(self) -> Path:
        return self.p.project.root / f".upload-{uuid.uuid4().hex[:8]}.zip"

    def _records(self) -> Dict[str, Any]:
        import json

        try:
            r = json.loads((self.ds / UPLOADS_FILE).read_text(encoding="utf-8"))
            return r if isinstance(r, dict) else {}
        except (OSError, ValueError):
            return {}

    def _write_records(self, r: Dict[str, Any]) -> None:
        import json
        import os

        tmp = self.ds / (UPLOADS_FILE + ".tmp")
        tmp.write_text(json.dumps(r), encoding="utf-8")
        os.replace(tmp, self.ds / UPLOADS_FILE)

    def _image_files(self) -> List[str]:
        imgs = self.ds / "images"
        return sorted(f"images/{p.name}" for p in imgs.iterdir() if p.is_file()) if imgs.is_dir() else []

    def remember_images(self, dataset_id: str) -> None:
        """The worker's dataset ``dataset_id`` has this project's images/ (training on it reuses them too)."""
        r = self._records()
        old = r.get(self.d.base_url)
        if isinstance(old, dict) and old.get("datasetId") and old["datasetId"] != dataset_id:
            try:
                self.client.delete_dataset(old["datasetId"])
            except Exception:  # noqa: BLE001
                pass
        r[self.d.base_url] = {"datasetId": dataset_id, "files": _signature(self.ds, self._image_files()),
                              "at": time.time()}
        self._write_records(r)

    def stored_images(self) -> Optional[str]:
        rec = self._records().get(self.d.base_url)
        if not isinstance(rec, dict) or not isinstance(rec.get("datasetId"), str):
            return None
        sig = _signature(self.ds, self._image_files())
        files = rec.get("files") or {}
        if not sig or any(files.get(k) != v for k, v in sig.items()):
            return None
        try:
            if rec["datasetId"] not in {x.get("datasetId") for x in self.client.datasets()}:
                return None
        except ApiError:
            return None
        return rec["datasetId"]

    def images_store(self) -> str:
        """The worker's copy of images/: the remembered one if it is still there and unchanged, else uploaded."""
        did = self.stored_images()
        if did is not None:
            self.p._log(f"{self.name} still has the photos: not sending them again")
            return did
        files = self._image_files()
        tmp = self._tmp()
        try:
            size = write_zip(self.ds, files, tmp, check=self.p._check)
            self.p._log(f"uploading {len(files)} photos to {self.name} ({size / 2**20:.0f} MB)")
            did = self.client.upload_dataset(tmp, self._progress("the photos"))
        finally:
            tmp.unlink(missing_ok=True)
        self.remember_images(did)
        return did

    def _dataset_for(self, extra: Sequence[str]) -> str:
        """The stored images plus ``extra`` (folders / files of the project) as one dataset on the worker."""
        store = self.images_store()
        files: List[str] = []
        for rel in extra:
            p = self.ds / rel
            if p.is_dir():
                files += [str(f.relative_to(self.ds)) for f in sorted(p.rglob("*")) if f.is_file()]
            elif p.is_file():
                files.append(rel)
        if not files:
            return store
        tmp = self._tmp()
        try:
            write_zip(self.ds, files, tmp, check=self.p._check)
            delta = self.client.upload_dataset(tmp, self._progress("masks and settings"))
        finally:
            tmp.unlink(missing_ok=True)
        self._cleanup.append(delta)
        try:
            did = self.client.compose([{"datasetId": store}, {"datasetId": delta}])["datasetId"]
        except ApiError as e:
            if e.status != 404:
                raise
            self.forget_images()                 # the photos are gone from the worker: send them again
            store = self.images_store()
            did = self.client.compose([{"datasetId": store}, {"datasetId": delta}])["datasetId"]
        self._cleanup.append(did)
        return did

    def forget_images(self) -> None:
        r = self._records()
        r.pop(self.d.base_url, None)
        self._write_records(r)

    def _run_job(self, dataset_id: str, request: Dict[str, Any], title: str) -> Dict[str, Any]:
        self.p._log(f"request on {self.name}: {request}")
        job = self.client.process(dataset_id, request, f"{self.p.project.name}: {title}", 0)
        self.p.track_remote("prep", self.client, job)
        try:
            return self.p._wait(job, client=self.client, where=self.name)
        finally:
            self.p.track_remote("prep", self.client, None)

    def _fetch(self, job: str, dest: Path, skip: Optional[str]) -> None:
        tmp = self.p.project.root / f".download-{uuid.uuid4().hex[:8]}.zip"
        q = f"?skip={quote(skip)}" if skip else ""
        self.p._progress(0.0, f"Downloading from {self.name}")
        try:
            self.client.download(f"/v1/jobs/{job}/output.zip{q}", tmp)
            merge_zip(tmp, dest)
        finally:
            tmp.unlink(missing_ok=True)

    def cleanup(self) -> None:
        for did in self._cleanup:
            try:
                self.client.delete_dataset(did)
            except Exception:  # noqa: BLE001
                pass
        self._cleanup.clear()

    # ------------------------------------------------------------ steps
    def frames(self, video: Path, request: Dict[str, Any], dest: Path) -> Dict[str, Any]:
        """Split a video on the worker: it goes up unchanged; the frames come back, and stay there as the stored
        copy of images/."""
        if "frames" not in self.caps:
            raise StepUnavailable(f"{self.name} can't split videos")
        if (request.get("stereo") or {}).get("rightVideo"):
            raise StepUnavailable("a separate right-eye video is split on this PC")
        self.p._log(f"uploading the video to {self.name} ({video.stat().st_size / 2**20:.0f} MB)")
        vid = self.client.upload(f"/v1/datasets/file?name={quote(request['video'])}", video,
                                 self._progress("the video"), ctype="application/octet-stream")["datasetId"]
        self._cleanup.append(vid)
        s = self._run_job(vid, request, "frames")
        self._fetch(s["jobId"], dest, None)
        out = s.get("outputDatasetId")
        if out and "chain" in self.caps:
            self.remember_images(out)            # the worker keeps the frames: masks / cameras don't resend them
        return s

    def masks(self, request: Dict[str, Any], dest: Path) -> Dict[str, Any]:
        model = request.get("maskModel")
        if not self.has_model(model):
            raise StepUnavailable(f"{self.name} doesn't have {model}")
        did = self._dataset_for([])
        s = self._run_job(did, request, f"masks {model}")
        self._fetch(s["jobId"], dest, "images")
        return s

    def solve(self, request: Dict[str, Any], dest: Path) -> Dict[str, Any]:
        if not self.has_model(request.get("features")):
            raise StepUnavailable(f"{self.name} doesn't have {request.get('features')}")
        extra = [x for x in (request.get("masksDir"), "splatprep-sky", request.get("priorPoses"), "frames.json")
                 if x]
        did = self._dataset_for(extra)
        s = self._run_job(did, request, "camera solve")
        self._fetch(s["jobId"], dest, None)      # the solve may undistort the images: take them as it left them
        return s

    # ------------------------------------------------------------ the whole pipeline as one job
    def pipeline_spec(self, video_request: Optional[Dict[str, Any]], has_priors: bool) -> Dict[str, Any]:
        """PROTOCOL.md "Whole pipeline jobs": every stage this project's options ask for."""
        o = self.p.o
        spec: Dict[str, Any] = {"staticRegions": bool(o.mask_static), "fisheyeMasks": True, "level": False,
                                "holdoutInterval": int(o.holdout_every) if o.holdout_every and o.holdout_every > 1
                                else None}
        if video_request is not None:
            spec["frames"] = {"request": video_request}
        models = [m for m in o.mask_models if m and m != "none"]
        if models:
            spec["masks"] = {"request": o.mask_request(models[0], "_masks"), "dilatePixels": 4}
        if o.sky:
            spec["sky"] = {"request": o.sky_request()}
        sfm = o.sfm_request(False, False, has_priors)
        for k in ("imagesDir", "outputDir", "masksDir", "skyMasksDir"):
            sfm.pop(k, None)
        spec["sfm"] = {"request": sfm, "sequential": o.source_kind in ("video", "video360"),
                       "enforceRig": o.source_kind in ("video360", "photos360")}
        spec["train"] = {"config": {**o.train_config(), "depthWeight": 0.0}}
        return spec

    def start_pipeline(self, video: Optional[Path], video_request: Optional[Dict[str, Any]],
                       has_priors: bool) -> str:
        """Upload the video (or the frames ingested here) and start the pipeline job; it is recorded in
        project.json (``pipelineJob``) so preparing again re-attaches to it."""
        if video is not None:
            self.p._log(f"uploading the video to {self.name} ({video.stat().st_size / 2**20:.0f} MB)")
            did = self.client.upload(f"/v1/datasets/file?name={quote(video_request['video'])}", video,
                                     self._progress("the video"), ctype="application/octet-stream")["datasetId"]
        else:
            did = self._dataset_for(["frames.json", "masks", "priors.json"])
        spec = self.pipeline_spec(video_request, has_priors)
        self.p._log(f"whole pipeline on {self.name}: {json.dumps(spec)}")
        job = self.client.post("/v1/pipeline", {"datasetId": did, "name": self.p.project.name,
                                                "pipeline": spec})["jobId"]
        self.p.project.update(pipelineJob={"jobId": job, "device": self.d.id, "worker": self.name,
                                           "options": options_fingerprint(self.p.o),
                                           "videoOnWorker": video is not None})
        return job

    def follow_pipeline(self, job: str, video_on_worker: bool) -> Path:
        """Follow the pipeline job: its stages show as this run's stages; once the dataset is prepared it is
        fetched into the project (the images too when the video was split there); then the training's previews
        and, at the end, the result. Returns the result's path."""
        from .compute import ComputeRun

        p = self.p
        run = ComputeRun(p, getattr(p, "devices", None), {})
        state = {"stage": None, "adopted": False}

        def on_status(s: Dict[str, Any]) -> None:
            p.emit("job", s)
            stage = s.get("stage")
            if stage and stage != state["stage"]:
                state["stage"] = stage
                title = STAGE_TITLES.get(stage)
                if stage == "frames":
                    title = "Extracting frames"
                elif stage == "masks":
                    title = next((t for t in p._titles if t.startswith("Masks")), None)
                if title in p._titles:
                    p._stage(title)
                p._log(f"on {self.name}: {stage}")
            total = s.get("totalSteps") or 0
            msg = s.get("message") or stage or ""
            p._progress(min(1.0, (s.get("step") or 0) / total) if total else 0.0, f"On {self.name}: {msg}")
            if not state["adopted"] and s.get("outputDatasetId") and (stage == "train" or s.get("state") == "completed"):
                self.adopt_dataset(s["outputDatasetId"], video_on_worker)
                state["adopted"] = True
        dest = p.project.new_output_path()
        out, _sky = run._finish_job(self.client, self.d, job, p.project.name, dest, on_status, with_preview=True,
                                    local=False, max_offline=PIPELINE_OFFLINE)
        p.project.update(pipelineJob=None, lastJob={"jobId": job, "device": self.name, "startedAt": time.time()})
        return out

    def adopt_dataset(self, dataset_id: str, video_on_worker: bool) -> None:
        """The prepared dataset into the project (without the images when they are already here)."""
        import shutil

        tmp = self.p.project.root / f".download-{uuid.uuid4().hex[:8]}.zip"
        q = "" if video_on_worker else "?skip=images"
        self.p._log(f"downloading the prepared dataset from {self.name}")
        try:
            self.client.download(f"/v1/datasets/{dataset_id}/archive.zip{q}", tmp)
            merge_zip(tmp, self.ds)
        finally:
            tmp.unlink(missing_ok=True)
        shutil.rmtree(self.ds / ".pipeline", ignore_errors=True)
        try:
            doc = json.loads((self.ds / "dataset.json").read_text(encoding="utf-8"))
            cams = doc.get("cameras") or []
            stats = {"cameras": len(cams), "masked": sum(1 for c in cams if c.get("mask")),
                     "sky": sum(1 for c in cams if c.get("sky")), "holdout": sum(1 for c in cams if c.get("holdout"))}
            self.p.project.update(datasetStats=stats)
            self.p._log(f"dataset: {stats}")
        except (OSError, ValueError):
            pass


def merge_zip(zip_path: Path, dest: Path, max_files: int = 2_000_000) -> int:
    """Unpack ``zip_path`` into ``dest``, replacing files of the same name (as the local export does). Names are
    checked like safezip's: relative, no "..", no links."""
    import shutil
    import stat

    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    root = dest.resolve()
    n = 0
    with zipfile.ZipFile(zip_path) as zf:
        infos = zf.infolist()
        if len(infos) > max_files:
            raise RuntimeError(f"the worker's output has {len(infos)} files")
        for info in infos:
            name = PurePosixPath(info.filename)
            if name.is_absolute() or ".." in name.parts or not name.parts:
                raise RuntimeError(f"unsafe name in the worker's output: {info.filename!r}")
            mode = (info.external_attr >> 16) & 0xFFFF
            if mode and stat.S_ISLNK(mode):
                raise RuntimeError(f"link in the worker's output: {info.filename!r}")
            target = (dest / Path(*name.parts)).resolve()
            if root not in target.parents and target != root:
                raise RuntimeError(f"unsafe name in the worker's output: {info.filename!r}")
            if info.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            tmp = target.with_name(target.name + ".part")
            with zf.open(info) as src, open(tmp, "wb") as out:
                shutil.copyfileobj(src, out, 1 << 20)
            if target.exists() or target.is_symlink():
                target.unlink()                  # a hard link into the source: replace, never write through it
            tmp.rename(target)
            n += 1
    return n
