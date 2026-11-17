# Development status

Where SplatWork stands, what is tested and how, what is known to be missing, and how releases are made. Last
release **1.3.0** (2026-11-16): SplatWork 1.3.0 for Mac, SplatWork for Linux 1.7.10, the first SplatWork for
Windows (1.3.0), SplatWork Capture 1.2 (unchanged). Changes after it go in the CHANGELOG's Unreleased section.

## Components and versions

| Component | Version | Where the version lives | Shipped as |
|---|---|---|---|
| SplatWork (Mac) | 1.3.0 | the release tag (`SPLAT_VERSION` in `mac/scripts/build_app.sh`); build number = commit count | `.dmg` on the GitHub release; a rolling `dev` pre-release from `main` |
| SplatWork for Linux | 1.7.10 | `linux/splatworker/__init__.py`, `linux/pyproject.toml`, `<release>` in `linux/flatpak/com.example.SplatWorker.metainfo.xml` | GPG-signed Flatpak repository on GitHub Pages (every push to `main`; key in the `FLATPAK_GPG_PRIVATE_KEY` secret), `.flatpak` bundle on the release |
| SplatWork Capture (iPhone) | 1.2 (build 2) | `MARKETING_VERSION` / `CURRENT_PROJECT_VERSION` in `ios/project.yml` | build from source (not on the App Store / TestFlight) |
| SplatWork for Windows | 1.3.0 | `windows/VERSION` (the release tag in releases); `bundle.json` in the app folder | unsigned `.msi` + portable `.zip` on the GitHub release |
| splatprep | 0.1.0 | `tools/pyproject.toml` | inside the Mac app and the Linux Flatpak |

Identifiers that must not change (installs and data depend on them): Flatpak app id `com.example.SplatWorker`, Bonjour
services `_splatworker._tcp` / `_splatcapture._tcp`, the Mac module and binary `SplatStudio`, Keychain service
`SplatStudio.WorkerToken`.

## What is tested, and how

| Area | How | Coverage |
|---|---|---|
| Linux worker + studio core | `cd linux && python -m pytest -q tests` (no GPU needed; fake backends) | 434 passed + 40 skipped (2026-11-16): API, GPU slots and job migration, trainer comparison, update checks, queue, pause/resume/shutdown, processing, datasets, devices, block training, GUI logic, formats, photo import, home video options |
| splatprep | `cd tools && python -m pytest -q tests` (needs opencv-python-headless, scipy; PyAV for the home-video end-to-end test) | 147 tests (2026-11-16), incl. GPS-neighbour pairs, GPS-held re-solve, take-off shots, home video restoration on synthetic tape artefacts |
| Vulkan trainer (`linux/vktrain`, stages 1, 2a, 2b; in the worker and app since 1.7.0) | `vktrain-selftest` and `tests/test_vktrain.py` on Mesa lavapipe in `linux/vktrain/ci/Dockerfile` (CI: `vulkan-trainer.yml`; the pytest file skips without glslang + a Vulkan device) | every kernel against a double-precision CPU reference and finite differences (incl. the 2b image-space loss pipeline: skybox, colour correction, decoupled SSIM, depth / normal losses; transient masks against trainer.py's rule; the camera-pose gradient; compact Adam against a host reference); the host strategies; end-to-end `splatwork-vk-train train` runs on synthetic scenes (both strategies, pause / resume incl. the 2b state, checkpoint, cancel, region, and one run per 2b feature on a scene made for it: sky, per-photo colour shifts, depth maps, moving objects, noisy poses). The worker side (`splatworker/vktrain.py`: trainer choice, job runner, `/v1/info`, setup wording) in `tests/test_vktrain_worker.py` without Vulkan, and through the real program in `test_vktrain.py`. Real GPUs: first runs on 2026-11-16 on BensPC (RTX 4080 and the integrated Radeon 610M training two blocks in parallel). *Compare trainers* (CUDA vs Vulkan on one GPU) and the RX 6700 XT test (`docs/dev/AMD_TEST_STEAMOS.md`) not run yet |
| Linux GTK interface | `linux/scripts/gui_smoke.py` under Xvfb in a Debian container with Mesa (OpenGL via llvmpipe): screenshots of every page | layout and wiring, not real GPUs |
| Mac app | `swift build -c release`; headless CLI: `--prepare`, `--train [--blocks N]`, `--eval`, `--convert`, `--render`, `--survey`; UI snapshots with `SPLAT_UI_SNAPSHOTS=<dir>` (plus `SPLAT_UI_PROJECT`, `SPLAT_UI_DRONE`) | end to end on real photos; the trainer's gradients are checked |
| iPhone app | `xcodebuild` for the simulator and for devices (Release) | builds; not run on a phone for 1.2 |
| SplatWork for Windows | `.github/workflows/windows.yml` on `windows-latest` (no GPU): the Linux suite and splatprep tests on Windows, worker smoke (start, `/v1/info`, mDNS, stop), GTK studio smoke with screenshots (Mesa software OpenGL), MSI build → silent install → installed worker + studio → firewall rules → uninstall | startup, packaging and discovery on one machine; not real GPUs, real drivers or a real LAN |
| Releases | `gh workflow run release.yml -f version=X.Y.Z-rc1` builds everything without publishing | 1.2.0 and 1.3.0 were dry-run first |

Not covered by automated tests: CUDA training on real NVIDIA GPUs (done by hand on an RTX 4080), Apple Object Capture
(flaky now and then: a set that fails once usually solves on the next run), multi-device runs on a real network.

## Known issues and gaps

- Cancelling a Linux multi-device run can leave the uploaded photos on the other workers (the delete arrives while
  the remote job is still cancelling).
- Not yet exercised on real hardware: Mac block training across several machines with Retry / Pause, the iPhone app
  (1.2), anything on Windows (CI only), RTX 50 cards.
- After an NVIDIA driver update the Flatpak's matching driver extension may be missing; the setup screen shows the
  `flatpak install flathub org.freedesktop.Platform.GL.nvidia-<version>//1.4` command.
- Drone video: locations from DJI flight logs (`.SRT`) aren't read yet.
- Home video (VHS): SplatWork for Linux takes the clips to use as numbers (the project log lists them); the Mac's
  clip list with pictures isn't mirrored there yet. Results depend heavily on the shot: see [VHS.md](VHS.md).
- The Vulkan trainer (default on AMD / Intel, opt-in on NVIDIA) is experimental: its first real-GPU runs were on
  2026-11-16 (RTX 4080 and Radeon 610M). A job moving from a slower to a faster GPU (1.7.8), *Compare trainers*
  (1.7.9) and the RX 6700 XT test (`AMD_TEST_STEAMOS.md`) haven't been confirmed on real hardware yet. Its
  densification still round-trips through the CPU. It isn't in the Windows app yet.
- The Mac's time left for a block run assumes one block at a time per device, so it reads high when a worker trains
  several blocks at once (one per GPU).
- SplatWork Capture's Stereo mode (wide + ultra-wide pairs) is built but not yet recorded on a real iPhone. Whether
  each phone delivers the lens calibration and depth maps alongside both photos is still to confirm. SplatWork for
  Linux doesn't import stereo bundles yet (Mac only).
- The colour-correction safeguard has not yet fired on real data (healthy runs stay quiet; the failure it targets
  was seen once, on a wrong camera solve).
- The Windows installer isn't code-signed, so SmartScreen warns on first run.
- The Mac-only additions of 2026-11-15 (preparation screen, Upload size, Leave out take-off shots switch) aren't in
  the Linux app's interface yet; splatprep's defaults apply there.
- The Mac release is ad-hoc signed (no Developer ID secrets in the repository), so it isn't notarized.
- Linux installs made before 2026-11-16 from the then unsigned repository keep updating, but without signature
  checks; a system-wide install from that time fails `flatpak update` ("Can't pull from untrusted non-gpg verified
  remote") until it is reinstalled per-user from the `.flatpakref` (linux/README.md).

## Next work

The plan is [ROADMAP.md](ROADMAP.md) (rewritten 2026-11-17 after a full review): rules for every change, milestones
in order, and a register of the recurring issues. In short:

0. **Measure:** a quality and speed benchmark with committed baselines, and a divergence guard in every trainer.
1. **Stop the known quality losses:** the MCMC collapse above about 1 M splats, fog in exports, presets, resume
   state, remote / local solve differences, pose-refinement outliers, colour correction.
2. **One trainer specification** across Metal, gsplat and Vulkan, with tests; a Mac test target.
3. **Quality from current research**, 4. **Speed**, 5. **Networked processing** (block split and merge v2,
   transfers, transport security), 6. **Easy to use**, 7. **Hardening**, 8. **Distribution and beta**.

New platforms, sources and trainers are frozen until milestones 0–2 are done.

## Releasing

1. Move the `## Unreleased` notes in `CHANGELOG.md` under `## X.Y.Z — date` (the release workflow puts that section
   first in the release notes).
2. Bump the Linux version (`__init__.py`, `pyproject.toml`, a `<release>` in the metainfo), `windows/VERSION`, and
   the iPhone version if they changed. Check that the latest Flatpak publish signed the repository (the live
   `.flatpakref` has a `GPGKey=` line).
3. Run the tests above, then a dry run: `gh workflow run release.yml -f version=X.Y.Z-rc1`.
4. Tag the commit the dry run built and push the tag:
   `git tag -a vX.Y.Z <commit> -m "SplatWork X.Y.Z" && git push origin vX.Y.Z`.
   The workflow builds the `.dmg`, the Flatpak bundle and the Windows `.msi` / `.zip` (via `windows.yml`), writes `SHA256SUMS.txt` and publishes the release; Mac apps
   offer the update, and the website's download button follows the latest release.

Every push to `main` also publishes the Mac developer build (`dev` pre-release) and, for changes under `linux/`,
`tools/`, `website/` or `docs/images/`, the Linux Flatpak repository and the website.
