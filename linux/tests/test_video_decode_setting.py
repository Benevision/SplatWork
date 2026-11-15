"""Settings → Training → Decode video on: reaches the worker and, through the environment, its job processes."""
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def test_setting_to_worker_to_jobs(tmp_path, monkeypatch):
    from splatworker.cli import build_parser
    from splatworker.gui.paths import AppPaths
    from splatworker.gui.settings import AppSettings

    paths = AppPaths(config_dir=tmp_path / "c", data_root=tmp_path / "d", cache_dir=tmp_path / "k")
    assert "--video-decode" not in AppSettings().serve_args(paths)
    args = build_parser().parse_args(["serve", *AppSettings(video_decode="cpu").serve_args(paths)])
    assert args.video_decode == "cpu"
    assert "video_decode" in AppSettings(video_decode="vaapi").field_errors()

    pytest.importorskip("fastapi")
    from conftest import FakeBackend
    from splatworker.config import ServerSettings
    from splatworker.server import create_app

    monkeypatch.delenv("SPLATPREP_VIDEO_DECODE", raising=False)
    create_app(ServerSettings(data_dir=tmp_path / "w", mdns=False, video_decode="cpu"), backend=FakeBackend(),
               devices=[None])
    assert os.environ["SPLATPREP_VIDEO_DECODE"] == "cpu"
    create_app(ServerSettings(data_dir=tmp_path / "w2", mdns=False), backend=FakeBackend(), devices=[None])
    assert os.environ["SPLATPREP_VIDEO_DECODE"] == "auto"
