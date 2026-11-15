# Development status

Where SplatWork stands, what is tested and how, what is known to be missing, and how releases are made. Last
release **1.2.0** (2026-11-13): SplatWork 1.2.0 for Mac, SplatWork for Linux 1.5.0, SplatWork Capture 1.2. Updated
2026-11-15 for what is on `main` since (see the CHANGELOG's Unreleased section): Linux 1.7.1, the Windows app, home
video, drone/GPS solving, the colour-correction safeguard, the Mac preparation screen, batched resumable uploads.

## Components and versions

| Component | Version | Where the version lives | Shipped as |
|---|---|---|---|
| SplatWork (Mac) | 1.2.0 | the release tag (`SPLAT_VERSION` in `mac/scripts/build_app.sh`); build number = commit count | `.dmg` on the GitHub release; a rolling `dev` pre-release from `main` |
| SplatWork for Linux | 1.7.1 (unreleased; 1.5.0 in the 1.2.0 release) | `linux/splatworker/__init__.py`, `linux/pyproject.toml`, `<release>` in `linux/flatpak/com.example.SplatWorker.metainfo.xml` | Flatpak repository on GitHub Pages (every push to `main`), `.flatpak` bundle on the release |
| SplatWork Capture (iPhone) | 1.2 (build 2) | `MARKETING_VERSION` / `CURRENT_PROJECT_VERSION` in `ios/project.yml` | build from source (not on the App Store / TestFlight) |
| SplatWork for Windows | 1.3.0 (unreleased; `windows/VERSION`, the release tag in releases) | `windows/VERSION`; `bundle.json` in the app folder | `.msi` + portable `.zip` on the GitHub release (from the next release) |
| splatprep | 0.1.0 | `tools/pyproject.toml` | inside the Mac app and the Linux Flatpak |

Identifiers that must not change (installs and data depend on them): Flatpak app id `com.example.SplatWorker`, Bonjour
services `_splatworker._tcp` / `_splatcapture._tcp`, the Mac module and binary `SplatStudio`, Keychain service
`SplatStudio.WorkerToken`.

## What is tested, and how

| Area | How | Coverage |
|---|---|---|
| Linux worker + studio core | `cd linux && python -m pytest -q tests` (no GPU needed; fake backends) | 362 passed + 25 skipped (2026-11-13): API, queue, pause/resume/shutdown, processing, datasets, devices, block training, GUI logic, formats, photo import, home video options |
| splatprep | `cd tools && python -m pytest -q tests` (needs opencv-python-headless, scipy; PyAV for the home-video end-to-end test) | 129 tests, incl. home video restoration on synthetic tape artefacts |
| Vulkan trainer (`linux/vktrain`, stages 1, 2a, 2b; in the worker and app since 1.7.0) | `vktrain-selftest` and `tests/test_vktrain.py` on Mesa lavapipe in `linux/vktrain/ci/Dockerfile` (CI: `vulkan-trainer.yml`; the pytest file skips without glslang + a Vulkan device) | every kernel against a double-precision CPU reference and finite differences (incl. the 2b image-space loss pipeline: skybox, colour correction, decoupled SSIM, depth / normal losses; transient masks against trainer.py's rule; the camera-pose gradient; compact Adam against a host reference); the host strategies; end-to-end `splatwork-vk-train train` runs on synthetic scenes (both strategies, pause / resume incl. the 2b state, checkpoint, cancel, region, and one run per 2b feature on a scene made for it: sky, per-photo colour shifts, depth maps, moving objects, noisy poses). The worker side (`splatworker/vktrain.py`: trainer choice, job runner, `/v1/info`, setup wording) in `tests/test_vktrain_worker.py` without Vulkan, and through the real program in `test_vktrain.py`. Against the CUDA trainer on the RTX 4080: by hand, `compare_training.sh` (not run yet); on AMD: `docs/dev/AMD_TEST_STEAMOS.md` step 4 (not run yet) |
| Linux GTK interface | `linux/scripts/gui_smoke.py` under Xvfb in a Debian container with Mesa (OpenGL via llvmpipe): screenshots of every page | layout and wiring, not real GPUs |
| Mac app | `swift build -c release`; headless CLI: `--prepare`, `--train [--blocks N]`, `--eval`, `--convert`, `--render`, `--survey`; UI snapshots with `SPLAT_UI_SNAPSHOTS=<dir>` (plus `SPLAT_UI_PROJECT`, `SPLAT_UI_DRONE`) | end to end on real photos; the trainer's gradients are checked |
| iPhone app | `xcodebuild` for the simulator and for devices (Release) | builds; not run on a phone for 1.2 |
| SplatWork for Windows | `.github/workflows/windows.yml` on `windows-latest` (no GPU): the Linux suite and splatprep tests on Windows, worker smoke (start, `/v1/info`, mDNS, stop), GTK studio smoke with screenshots (Mesa software OpenGL), MSI build → silent install → installed worker + studio → firewall rules → uninstall | startup, packaging and discovery on one machine; not real GPUs, real drivers or a real LAN |
| Releases | `gh workflow run release.yml -f version=X.Y.Z-rc1` builds everything without publishing | 1.2.0 was dry-run first |

Not covered by automated tests: CUDA training on real NVIDIA GPUs (done by hand on an RTX 4080), Apple Object Capture
(flaky now and then: a set that fails once usually solves on the next run), multi-device runs on a real network.

## Known issues and gaps

- Cancelling a Linux multi-device run can leave the uploaded photos on the other workers (the delete arrives while
  the remote job is still cancelling).
- Not yet exercised on real hardware for 1.2: Mac block training across several machines with Retry / Pause, the
  iPhone app, SplatWork for Linux 1.5.0 on an NVIDIA PC.
- After an NVIDIA driver update the Flatpak's matching driver extension may be missing; the setup screen shows the
  `flatpak install flathub org.freedesktop.Platform.GL.nvidia-<version>//1.4` command.
- Drone video: locations from DJI flight logs (`.SRT`) aren't read yet.
- Home video (VHS): SplatWork for Linux takes the clips to use as numbers (the project log lists them); the Mac's
  clip list with pictures isn't mirrored there yet. Results depend heavily on the shot: see [VHS.md](VHS.md).
- The Vulkan trainer (default on AMD / Intel, opt-in on NVIDIA) has only run on Mesa's software driver; the
  RTX 4080 comparison (`linux/vktrain/compare_training.sh`) and the RX 6700 XT test (`AMD_TEST_STEAMOS.md`) are
  still to do. Its densification still round-trips through the CPU.
- Drone surveys: a fast (GLOMAP) solve that matches GPS but places too few photos (498 of 1,229 on the sample) is
  kept as it is; the GPS-held re-solve, which placed 959, only runs when the GPS check fails.
- Training that should start on a Linux worker fails ("No Linux worker selected") if the worker is unreachable
  at that moment (e.g. updating itself) instead of waiting.
- The colour-correction safeguard has not yet fired on real data (healthy runs stay quiet; the failure it targets
  was seen once, on a wrong camera solve).
- The Windows gsplat 1.6 (CUDA 12.8) wheel for RTX 50 cards is still being built in CI; RTX 50 can't train on
  Windows until it lands. Nothing on Windows has run on real hardware.
- The Mac-only additions of 2026-11-15 (preparation screen, Upload size, Leave out take-off shots switch) aren't in
  the Linux app's interface yet; splatprep's defaults apply there.
- The Mac release is ad-hoc signed (no Developer ID secrets in the repository), so it isn't notarized.

## Next work

1. **Drone solving:** run the GPS-held re-solve when a GPS survey's fast solve places too few photos; let a
   worker-bound training start wait for the worker instead of failing.
2. **Real-GPU checks of the Vulkan trainer** (RTX 4080 comparison, RX 6700 XT on SteamOS), then GPU densification —
   see [GPU_SUPPORT.md](GPU_SUPPORT.md).
3. **Linux parity with the Mac:** the Projects list with stage checkpoints and resume, drone mode and the photo
   switches, Show Live View, block Retry / Leave Out / Pause, the new preparation screen and Upload size.
4. **Windows:** land the RTX 50 wheel; a first test on a real Windows PC with an NVIDIA GPU ([WINDOWS.md](WINDOWS.md)).
5. **Release 1.3.0** once the above settles (fresh dry run first).
6. **Steam Frame capture:** recording with the stereo cameras and the IR cameras together, for room meshing that
   makes splats more accurate.

## Releasing

1. Move the `## Unreleased` notes in `CHANGELOG.md` under `## X.Y.Z — date` (the release workflow puts that section
   first in the release notes).
2. Bump the Linux version (`__init__.py`, `pyproject.toml`, a `<release>` in the metainfo) and the iPhone version if
   they changed.
3. Run the tests above, then a dry run: `gh workflow run release.yml -f version=X.Y.Z-rc1`.
4. Tag the commit the dry run built and push the tag:
   `git tag -a vX.Y.Z <commit> -m "SplatWork X.Y.Z" && git push origin vX.Y.Z`.
   The workflow builds the `.dmg`, the Flatpak bundle and the Windows `.msi` / `.zip` (via `windows.yml`), writes `SHA256SUMS.txt` and publishes the release; Mac apps
   offer the update, and the website's download button follows the latest release.

Every push to `main` also publishes the Mac developer build (`dev` pre-release) and, for changes under `linux/`,
`tools/`, `website/` or `docs/images/`, the Linux Flatpak repository and the website.
