# Contributing to SplatWork

Thanks for your interest! Bug reports, ideas, documentation fixes and code are all welcome.

## Reporting bugs and asking for features

[Open an issue](https://github.com/Benevision/SplatWork/issues/new/choose). For a bug, say which app and version
(SplatWork → About, or SplatWork for Linux's Preferences), what you did, what you expected and what happened. A project's
`prep.log`, the training dashboard and a screenshot help a lot. Security problems go through
[SECURITY.md](SECURITY.md), not public issues.

## Repository layout

| Folder | Contents | Language |
|---|---|---|
| `mac/` | SplatWork for Mac: app, Metal renderer and trainer, pipeline, worker client (`Sources/SplatStudio`) | Swift 6, SwiftUI, Metal |
| `ios/` | SplatWork Capture: capture app plus the on-device trainer (`SplatTrainKit`) | Swift, ARKit |
| `linux/` | SplatWork for Linux: worker server (`splatworker/`), GTK studio (`splatworker/gui/`), Flatpak (`flatpak/`) | Python, FastAPI, GTK 4 |
| `linux/vktrain/` | The Vulkan trainer for AMD, Intel and (by choice) NVIDIA GPUs: the `splatwork-vk-train` program and its self-test | C++17, GLSL compute shaders (compiled to SPIR-V) |
| `windows/` | SplatWork for Windows: launcher, bundle and MSI build scripts, smoke tests; the app itself is the shared `linux/` code | C, Python, WiX |
| `tools/` | splatprep processing toolkit and evaluation scripts | Python, PyTorch |
| `docs/` | User guide, settings reference, architecture, protocols, developer notes | Markdown |
| `website/` | The project website, published to GitHub Pages with the Flatpak repository | HTML, CSS, JavaScript |
| `branding/` | Logo and app icons, generated from the same geometry | SVG |
| `steamframe/` | Experimental headset capture prototype | Python |

[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) explains how the parts fit together, and
[docs/PROTOCOL.md](docs/PROTOCOL.md) defines the formats and the HTTP API they share.

## Building and testing

**Mac app** (Apple silicon, Xcode 16+):

```bash
cd mac
swift build -c release            # build the app and its command line
./scripts/build_app.sh            # make build/SplatWork.app
./scripts/update_local.sh         # build and install it, waiting until the app is closed
```

**iPhone app** (Xcode 26+, XcodeGen): `cd ios && xcodegen generate`, then build the *SplatCapture* scheme. Run
`xcodegen generate` again whenever you add or remove files; `project.yml` is the source of truth.

**SplatWork for Linux and splatprep** (Python 3.10+):

```bash
cd linux && python -m pytest -q    # worker, studio core and GUI logic (no GPU needed)
cd tools && python -m pytest -q    # splatprep
cd linux/flatpak && bash build.sh  # build and install the Flatpak locally
```

The GTK viewer can be tested without a display; see [docs/dev/LINUX_STUDIO.md](docs/dev/LINUX_STUDIO.md).

**Vulkan trainer** (`linux/vktrain`, CMake, Vulkan headers and loader, `glslangValidator`). Its kernel self-test
checks every kernel against a CPU reference and finite-difference gradients, and needs no GPU: the test image runs
it on Mesa's software Vulkan driver (lavapipe).

```bash
docker build -t splat-vk-test -f linux/vktrain/ci/Dockerfile linux/vktrain/ci
docker run --rm -v "$PWD":/w -w /w/linux/vktrain splat-vk-test \
  sh -c 'cmake -S . -B build -G Ninja && cmake --build build && ./build/vktrain-selftest'
```

`linux/tests/test_vktrain.py` runs training end to end on a synthetic scene; it is skipped when Vulkan isn't
installed. The *Vulkan trainer* workflow (`vulkan-trainer.yml`) runs both on every push that touches `linux/vktrain/`.

**SplatWork for Windows** is built and tested on GitHub Actions only (`windows.yml`, `windows-latest`, no GPU): the
Linux worker and studio suite, the splatprep suite, worker and studio smoke tests, and the MSI installed and
uninstalled on the runner. See [windows/README.md](windows/README.md).

**Settings.** Every user-facing setting is described once, in `mac/Sources/SplatStudio/Core/SettingsCatalog.swift`.
After changing it, regenerate the reference and the Linux catalog:

```bash
mac/.build/release/SplatStudio --dump-settings > docs/SETTINGS.md
python linux/scripts/gen_catalog.py
```

## Sending changes

1. Fork the repository and create a branch from `main`.
2. Keep each pull request to one change, with tests where it makes sense.
3. Run the tests above for the parts you touched, and say in the pull request what you checked by hand.
4. Write user-facing text plainly, and use the product names SplatWork, SplatWork Capture, SplatWork for Linux and
   SplatWork for Windows.

Every push to `main` builds the Mac developer build (`mac-dev.yml`), the Linux Flatpak repository and website
(`splat-worker-flatpak.yml`) and SplatWork for Windows (`windows.yml`), each when its files change, so `main` should
always be releasable. The Vulkan trainer (`vulkan-trainer.yml`) and the gsplat wheels for Windows
(`gsplat-windows-wheel.yml`) and AMD ROCm (`gsplat-rocm-wheel.yml`) have workflows of their own. Pushing a `v*`
tag runs `release.yml`.

## Licensing

SplatWork is licensed under the [Apache License 2.0](LICENSE). By contributing, you agree that your contributions
are licensed under the same terms. Don't copy code from projects with incompatible licences (for example GPL code);
reimplement ideas from papers and other projects in your own code and credit them in comments.
