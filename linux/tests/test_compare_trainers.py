"""Compare trainers (splatworker benchmark --compare-trainers, Models & Benchmarks → Compare trainers): the same
scene on the same GPU with gsplat and then the Vulkan trainer, as two ordinary worker jobs that each name their
trainer. Fake client / fake trainers."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

pytest.importorskip("numpy")

from splatworker.studio import benchmark as B  # noqa: E402

TINY = B.BenchmarkSpec(width=96, height=72, views=8, splats=2000, steps=40, measure_from=10)
SPEEDS = {"gsplat": 40.0, "vulkan": 16.0}
QUALITY = {"gsplat": (24.0, 0.80), "vulkan": (23.7, 0.79)}


class FakeClient:
    """A worker whose jobs advance ``SPEEDS[trainer]`` steps per (fake) second at each poll."""

    def __init__(self, trainers=("gsplat", "vulkan")):
        self.trainers = list(trainers)
        self.bodies = []
        self.jobs = {}
        self.deleted_datasets = []
        self.vram = 1000

    def info(self):
        return {"name": "BensPC", "workerVersion": "1.7.9", "trainers": self.trainers,
                "gpus": [{"name": "RTX 4080", "memoryMB": 16376, "freeMemoryMB": 16376 - self.vram}],
                "slots": [{"id": "cuda:0", "name": "NVIDIA GeForce RTX 4080", "cuda": 0, "vulkan": 0},
                          {"id": "vk:0000:11:00.0", "name": "AMD Radeon 610M", "cuda": None, "vulkan": 1}]}

    def local_dataset(self, path):
        return {"datasetId": "dscene"}

    def delete_dataset(self, did):
        self.deleted_datasets.append(did)

    def post(self, path, body):
        assert path == "/v1/jobs"
        self.bodies.append(body)
        jid = f"j{len(self.bodies)}"
        self.jobs[jid] = {"trainer": body["trainer"], "t": 0, "steps": body["config"]["iterations"]}
        return {"jobId": jid}

    def job(self, jid):
        j = self.jobs[jid]
        j["t"] += 1
        self.vram = 1000 + (6000 if j["trainer"] == "gsplat" else 4000)
        step = min(j["steps"], int(j["t"] * SPEEDS[j["trainer"]] / 4))
        done = step >= j["steps"]
        if done:
            self.vram = 1000
        psnr, ssim = QUALITY[j["trainer"]]
        return {"jobId": jid, "state": "completed" if done else "training", "step": step,
                "elapsedSeconds": j["t"] / 4, "trainer": j["trainer"], "numGaussians": 2000,
                "evalPsnr": psnr if done else None, "evalSsim": ssim if done else None, "psnr": psnr}

    def log(self, jid):
        return f"log of {jid}"

    def delete(self, path):
        pass


def test_compare_runs_both_trainers_on_one_gpu(tmp_path):
    c = FakeClient()
    rec = B.compare_trainers(c, tmp_path / "scene", TINY, local=True, quick=True, poll=0)
    assert [b["trainer"] for b in c.bodies] == ["gsplat", "vulkan"]
    assert all(b["slot"] == "cuda:0" for b in c.bodies)                    # the GPU both trainers can use
    assert all(b["config"]["strategy"] == "mcmc" and b["config"]["evalInterval"] == TINY.steps
               and b["config"]["seed"] == 7 for b in c.bodies)
    g, v = rec["runs"]
    assert g["stepsPerSecond"] == pytest.approx(40.0, rel=0.05) and v["stepsPerSecond"] == pytest.approx(16.0, rel=0.05)
    assert g["peakVramMB"] == 6000 and v["peakVramMB"] == 4000
    assert (g["evalPsnr"], v["evalSsim"], g["numGaussians"]) == (24.0, 0.79, 2000)
    cmp = rec["comparisons"][0]
    assert cmp["speedRatio"] == pytest.approx(0.4, rel=0.05) and cmp["psnrDiff"] == pytest.approx(-0.3)
    assert cmp["summary"] == "Vulkan (mcmc): 0.40× the speed, -0.30 dB, SSIM -0.010"
    assert rec["gpu"] == "NVIDIA GeForce RTX 4080" and rec["size"] == "quick"
    assert c.deleted_datasets == ["dscene"]                                 # registered once, removed once

    path = B.save_comparison(tmp_path / "cfg", rec)
    saved = B.load_comparisons(tmp_path / "cfg")
    assert path.name == B.COMPARE_FILE and saved[0]["runs"][0]["logFile"].endswith("-gsplat-mcmc.log")
    assert "log" not in saved[0]["runs"][0]
    assert (tmp_path / "cfg" / B.COMPARE_LOGS / saved[0]["runs"][1]["logFile"]).read_text() == "log of j2"

    from splatworker.cli import compare_table
    table = compare_table(rec)
    assert "gsplat   mcmc" in table and "Vulkan (mcmc): 0.40× the speed" in table


def test_both_strategies_and_missing_trainers(tmp_path):
    c = FakeClient()
    rec = B.compare_trainers(c, tmp_path / "scene", TINY, strategies=("mcmc", "adaptive"), poll=0)
    assert [(r["trainer"], r["strategy"]) for r in rec["runs"]] == [
        ("gsplat", "mcmc"), ("vulkan", "mcmc"), ("gsplat", "adaptive"), ("vulkan", "adaptive")]
    assert [x["strategy"] for x in rec["comparisons"]] == ["mcmc", "adaptive"]
    with pytest.raises(B.BenchmarkError, match="gsplat .*isn't available"):
        B.compare_trainers(FakeClient(trainers=["vulkan"]), tmp_path / "scene", TINY, poll=0)
    with pytest.raises(B.BenchmarkError, match="the Vulkan trainer isn't available"):
        B.compare_trainers(FakeClient(trainers=["gsplat"]), tmp_path / "scene", TINY, poll=0)


def test_worker_takes_a_trainer_per_job(tmp_path, monkeypatch):
    pytest.importorskip("fastapi")
    from conftest import make_dataset, zip_dir
    from test_api import AUTH, make_client, upload, wait_state

    from splatworker import vktrain

    monkeypatch.setattr(vktrain, "available_trainers", lambda: ["gsplat", "vulkan"])
    c, backend = make_client(tmp_path)
    with c:
        ds = upload(c, zip_dir(make_dataset(tmp_path / "K"))).json()["datasetId"]
        r = c.post("/v1/jobs", headers=AUTH, json={"datasetId": ds, "trainer": "vulkan"})
        assert r.status_code == 200, r.text
        s = wait_state(c, r.json()["jobId"], ("completed", "failed"))
        assert s["trainer"] == "vulkan" and backend.contexts[-1].trainer == "vulkan"
        assert c.post("/v1/jobs", headers=AUTH, json={"datasetId": ds, "trainer": "cuda"}).status_code == 400
        monkeypatch.setattr(vktrain, "available_trainers", lambda: ["gsplat"])
        r = c.post("/v1/jobs", headers=AUTH, json={"datasetId": ds, "trainer": "vulkan"})
        assert r.status_code == 400 and "isn't available" in r.json()["error"]
        assert c.get("/v1/info", headers=AUTH).json()["trainers"] == ["gsplat"]
