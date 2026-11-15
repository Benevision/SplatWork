"""Check for Updates… (the menu item and Settings row): the manual check and what it offers."""
from __future__ import annotations

import io
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from splatworker.gui import updates as U  # noqa: E402
from splatworker.gui import windows_update as W  # noqa: E402


def _url(tmp_path, monkeypatch, data):
    f = tmp_path / "version.json"
    f.write_text(json.dumps(data) if not isinstance(data, str) else data)
    monkeypatch.setenv("SPLATWORKER_UPDATE_URL", f.as_uri())


def test_check_now_tells_current_from_newer_from_failure(tmp_path, monkeypatch):
    _url(tmp_path, monkeypatch, {"version": "1.7.9", "commit": "abc"})
    r = U.check_now(current="1.7.5")
    assert r["status"] == "newer" and r["info"]["version"] == "1.7.9" and r["latest"] == "1.7.9"
    assert U.check(timeout=2) is None or U.check(timeout=2)["version"] == "1.7.9"
    _url(tmp_path, monkeypatch, {"version": "1.7.5"})
    assert U.check_now(current="1.7.5") == {"status": "current", "info": None, "latest": "1.7.5", "error": ""}
    monkeypatch.setenv("SPLATWORKER_UPDATE_URL", (tmp_path / "missing.json").as_uri())
    r = U.check_now(current="1.7.5")
    assert r["status"] == "error" and r["error"]
    _url(tmp_path, monkeypatch, "not json")
    assert U.check_now(current="1.7.5")["status"] == "error"
    monkeypatch.setenv("SPLATWORKER_UPDATE_URL", "")
    assert U.check_now()["status"] == "off"


def test_what_the_dialog_offers():
    newer = {"status": "newer", "info": {"version": "1.7.9"}, "latest": "1.7.9", "error": ""}
    v = U.manual_view(newer, flatpak=True, busy=0, current="1.7.5")
    assert v["heading"] == "SplatWork for Linux 1.7.9 is available" and v["default"] == "update"
    assert [r[0] for r in v["responses"]] == ["close", "update"]
    v = U.manual_view(newer, flatpak=True, busy=2, current="1.7.5")
    assert [r[0] for r in v["responses"]] == ["close", "now", "after"] and v["default"] == "after"
    assert "2 job(s) are running or queued" in v["body"]
    v = U.manual_view({"status": "current", "latest": "1.7.5"}, flatpak=True, busy=0, current="1.7.5")
    assert v["heading"] == "Up to date" and v["body"] == "SplatWork for Linux 1.7.5 is up to date."
    v = U.manual_view({"status": "error", "error": "Name or service not known"}, flatpak=True, busy=0)
    assert v["body"].startswith("Couldn't check: Name or service not known") and v["responses"][0][0] == "close"
    assert U.manual_view({"status": "off"}, flatpak=True, busy=0)["heading"] == "Update checks are off"
    # not a Flatpak install (a source build): the version and the download page
    v = U.manual_view(newer, flatpak=False, busy=0, current="1.7.5")
    assert [r[0] for r in v["responses"]] == ["close", "open"] and v["page"] == U.RELEASES_PAGE
    assert U.UPDATE_COMMAND in v["body"]
    # SplatWork for Windows: its own MSI updater
    win = {"status": "newer", "info": {"version": "1.8.0", "msi_url": "https://x/a.msi"}}
    v = U.manual_view(win, flatpak=False, busy=1, product="SplatWork for Windows")
    assert [r[0] for r in v["responses"]] == ["close", "windows"] and "Windows 1.8.0" in v["heading"]


class _Resp(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def test_windows_check_now(monkeypatch):
    rel = {"tag_name": "v1.8.0", "assets": [{"name": "SplatWork-1.8.0-Windows-x64.msi",
                                              "browser_download_url": "https://x/a.msi"}], "html_url": "https://p"}
    r = W.check_now("1.7.0", opener=lambda req, timeout: _Resp(json.dumps(rel).encode()))
    assert r["status"] == "newer" and r["info"]["msi_url"] == "https://x/a.msi"
    assert W.check_now("1.8.0", opener=lambda req, timeout: _Resp(json.dumps(rel).encode()))["status"] == "current"

    def boom(req, timeout):
        raise OSError("offline")
    assert W.check_now("1.7.0", opener=boom) == {"status": "error", "info": None, "latest": None, "error": "offline"}


def test_update_after_jobs_rule():
    # "Update After Jobs" makes the auto-update rule apply once, with automatic updates off
    assert U.next_action(U.AVAILABLE, False, False) is None
    assert U.next_action(U.AVAILABLE, True, True) is None          # still waits for the jobs
    assert U.next_action(U.INSTALLED, True, False) == "restart"
