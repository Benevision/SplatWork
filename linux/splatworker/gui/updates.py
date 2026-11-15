"""Update check against the Flatpak repository published by GitHub Actions.

The workflow (``.github/workflows/splat-worker-flatpak.yml``) publishes, next to the OSTree repo,
``version.json`` = ``{"version": "1.1.0", "commit": "...", "date": "..."}``. The app compares it with
its own version. Inside the Flatpak it then updates itself through the Flatpak portal
(``org.freedesktop.portal.Flatpak``: an update monitor + ``Update``, see :mod:`.gtk.flatpak_update`) and
restarts into the new version with ``Spawn(LATEST_VERSION)`` once no job is running — the same thing GNOME
Software would do, without leaving the app. Outside the Flatpak it only tells you. Set
``SPLATWORKER_UPDATE_URL`` to override the URL, or to an empty string to disable checks.

The pure decisions live here (tested without GTK / D-Bus).
"""
from __future__ import annotations

import json
import os
import re
import urllib.request
from typing import Iterable, List, Optional, Tuple

from .. import __version__

DEFAULT_URL = "https://benevision.github.io/SplatWork/flatpak/version.json"
RELEASES_PAGE = "https://github.com/Benevision/SplatWork/releases"
PRODUCT = "SplatWork for Linux"
APP_ID = "com.example.SplatWorker"
UPDATE_COMMAND = f"flatpak update --user {APP_ID}"


def update_url() -> str:
    return os.environ.get("SPLATWORKER_UPDATE_URL", DEFAULT_URL)


def parse_version(v: str) -> Tuple[int, ...]:
    return tuple(int(x) for x in re.findall(r"\d+", v)[:4]) or (0,)


def is_newer(remote: str, local: str = __version__) -> bool:
    return parse_version(remote) > parse_version(local)


def check(timeout: float = 6.0) -> Optional[dict]:
    """The published version info if it is newer than this app, else None (also on any error)."""
    r = check_now(timeout)
    return r["info"] if r["status"] == "newer" else None


def check_now(timeout: float = 8.0, current: str = __version__) -> dict:
    """The manual check (Check for Updates…): {"status": "newer" | "current" | "error" | "off", "info", "latest",
    "error"} — unlike :func:`check` it tells "up to date" from "couldn't check"."""
    url = update_url()
    if not url:
        return {"status": "off", "info": None, "latest": None, "error": ""}
    try:
        req = urllib.request.Request(url, headers={"User-Agent": f"SplatWorker/{current}", "Cache-Control": "no-cache"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            info = json.loads(r.read(64 * 1024).decode("utf-8"))
        v = str(info.get("version", ""))
        if not v:
            return {"status": "error", "info": None, "latest": None, "error": "the version file has no version"}
    except Exception as e:  # noqa: BLE001
        reason = getattr(e, "reason", None) or e
        return {"status": "error", "info": None, "latest": None, "error": str(reason) or type(e).__name__}
    newer = is_newer(v, current)
    return {"status": "newer" if newer else "current", "info": info if newer else None, "latest": v, "error": ""}


def manual_view(result: dict, *, flatpak: bool, busy: int, current: str = __version__,
                product: str = PRODUCT) -> dict:
    """What Check for Updates… shows: {"heading", "body", "responses": [(id, label, appearance)], "default"}.
    ``flatpak``: the portal can update this install; ``busy``: jobs running or queued (a restart stops them).
    Response ids: "close", "update" (download, install, restart), "after" (download now, restart when the jobs
    finish), "now" (stop the jobs and restart when installed), "windows" (the MSI updater), "open" (the
    download page)."""
    status = result.get("status")
    close = ("close", "Close", "default")
    if status == "off":
        return {"heading": "Update checks are off", "default": "close", "responses": [close],
                "body": "SPLATWORKER_UPDATE_URL is set to an empty value, so SplatWork doesn't look for updates."}
    if status == "error":
        return {"heading": "Couldn't check for updates", "default": "close", "responses": [close],
                "body": f"Couldn't check: {result.get('error') or 'unknown error'}. Check the internet connection "
                        f"and try again."}
    if status == "current":
        latest = result.get("latest") or current
        extra = "" if latest == current else f" (the newest published version is {latest})"
        return {"heading": "Up to date", "default": "close", "responses": [close],
                "body": f"{product} {current} is up to date{extra}."}
    info = result.get("info") or {}
    v = info.get("version") or result.get("latest") or ""
    head = f"{product} {v} is available"
    body = f"You have {current}."
    later = ("close", "Not Now", "default")
    if flatpak:
        if busy:
            return {"heading": head, "default": "after", "responses": [
                later, ("now", "Stop Jobs and Update", "destructive"), ("after", "Update After Jobs", "suggested")],
                "body": body + f" {busy} job(s) are running or queued. Update After Jobs downloads it now and "
                               "restarts SplatWork when they have finished; Stop Jobs and Update restarts as soon "
                               "as it is installed (paused training continues after the restart)."}
        return {"heading": head, "default": "update", "responses": [later, ("update", "Update and Restart", "suggested")],
                "body": body + " SplatWork downloads it, installs it and restarts into it."}
    if info.get("msi_url"):
        return {"heading": head, "default": "windows", "responses": [later, ("windows", "Update…", "suggested")],
                "body": body + " The installer is downloaded, checked and started; SplatWork closes for it."}
    page = info.get("page") or RELEASES_PAGE
    return {"heading": head, "default": "open", "page": page,
            "responses": [("close", "Close", "default"), ("open", "Open Download Page", "suggested")],
            "body": body + f" This copy isn't a Flatpak install, so it can't update itself: get the new version "
                           f"from {page}, or update the Flatpak with:\n\n{UPDATE_COMMAND}"}


# ------------------------------------------------------------------ self-update (Flatpak portal)
FINAL_STATES = ("completed", "failed", "cancelled")
# org.freedesktop.portal.Flatpak.UpdateMonitor Progress "status"
PROGRESS_RUNNING, PROGRESS_EMPTY, PROGRESS_DONE, PROGRESS_FAILED = 0, 1, 2, 3
SPAWN_LATEST_VERSION = 2     # FLATPAK_SPAWN_FLAGS_LATEST_VERSION

# Updater states (shown in the window banner)
IDLE, AVAILABLE, DOWNLOADING, INSTALLED, FAILED = "idle", "available", "downloading", "installed", "failed"


def busy_jobs(jobs: Iterable[dict]) -> List[dict]:
    """Jobs a restart would interrupt (queued ones too: they would restart from scratch)."""
    return [j for j in jobs or [] if j.get("state") not in FINAL_STATES]


def next_action(state: str, auto: bool, busy: bool) -> Optional[str]:
    """What the updater does on its own: "update" (download + install), "restart", or None (wait / ask)."""
    if not auto or busy:
        return None
    if state == AVAILABLE:
        return "update"
    if state == INSTALLED:
        return "restart"
    return None


def state_from_monitor(info: dict) -> Optional[str]:
    """UpdateAvailable(update_info): already deployed by someone else (needs only a restart) or downloadable."""
    running, local, remote = (info.get(k) for k in ("running-commit", "local-commit", "remote-commit"))
    if local and running and local != running:
        return INSTALLED
    if remote and running and remote != running:
        return AVAILABLE
    return None


def parse_progress(p: dict) -> Tuple[str, float, str]:
    """(state, overall fraction, error) from a Progress(info) signal."""
    status = int(p.get("status", PROGRESS_RUNNING))
    if status == PROGRESS_FAILED:
        return FAILED, 0.0, str(p.get("error_message") or p.get("error") or "the update failed")
    if status == PROGRESS_DONE:
        return INSTALLED, 1.0, ""
    if status == PROGRESS_EMPTY:
        return IDLE, 1.0, ""
    n_ops = max(1, int(p.get("n_ops", 1)))
    op = min(int(p.get("op", 0)), n_ops - 1)
    pct = max(0, min(100, int(p.get("progress", 0))))
    return DOWNLOADING, (op + pct / 100.0) / n_ops, ""


def restart_argv(background: bool) -> List[str]:
    """Command the portal starts (inside the new version's sandbox) to replace this instance."""
    return ["splatworker-app", "--after-update"] + (["--background"] if background else [])


def banner_text(state: str, version: str, auto: bool, busy: bool, fraction: float = 0.0,
                error: str = "") -> Tuple[str, Optional[str]]:
    """(title, button label or None) for the window banner."""
    name = f"SplatWork {version}".strip() if version else "A SplatWork update"
    if state == AVAILABLE:
        if auto and busy:
            return f"{name} is available. It installs and restarts when no jobs are running.", "Download Now"
        return f"{name} is available.", "Update"
    if state == DOWNLOADING:
        return f"Downloading {name}… {int(fraction * 100)}%", None
    if state == INSTALLED:
        if auto and busy:
            return f"{name} is installed. SplatWork restarts into it when no jobs are running.", "Restart Now"
        return f"{name} is installed.", "Restart"
    if state == FAILED:
        return f"Couldn't update automatically: {error}", "How to Update"
    return "", None
