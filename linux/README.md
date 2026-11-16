# SplatWork for Linux

SplatWork for Linux is the Linux member of the SplatWork suite. It does two jobs in one app:

- **A GPU worker.** SplatWork on the Mac (and SplatWork Capture on the iPhone) send it training and dataset
  processing jobs over the local network. On NVIDIA cards training uses
  [gsplat](https://github.com/nerfstudio-project/gsplat), the CUDA rasterizer from nerfstudio; on AMD and Intel
  cards it uses SplatWork's own Vulkan trainer (experimental, built into the app). Processing uses
  [splatprep](../tools/README.md) for video splitting, learned camera solving and masks.
- **A desktop studio.** A complete SplatWork for Linux: a viewer (PLY, compressed PLY, `.splat`, `.spz`, learned
  skybox), New Project (photos, video, 360° video and photos, stereo footage, iPhone captures, COLMAP, Nerfstudio),
  editing (brush, crop, thresholds, Auto clean, mask editor), export and turntable videos, an iPhone Inbox, and
  training on this PC or across several machines.

Everything GPU-heavy runs as jobs of one server with one job slot per GPU, integrated GPUs included, so work started
at the PC and work sent from a Mac share the GPUs fairly (the person at the PC goes first among waiting jobs; nothing
running is interrupted).
Projects are saved in `~/Documents/SplatWork`, in the same layout as the Mac, so a project opens on either.

```
SplatWork (Mac) ──Bonjour _splatworker._tcp──▶ discovers the worker
                 ──POST /v1/datasets (zip)────▶ datasets/<id>/
                 ──POST /v1/jobs {config}─────▶ queue ─▶ GPU slot ─▶ trainer process
                 ◀─GET /v1/jobs/{id} (poll)───   JobStatus (step, loss, psnr, numGaussians, eta)
                 ◀─GET preview.splat / result.ply
```

The API is documented in [docs/PROTOCOL.md](../docs/PROTOCOL.md) §2–§4, §6 and §7. How the studio is built:
[docs/dev/LINUX_STUDIO.md](../docs/dev/LINUX_STUDIO.md).

**Requirements:** a Linux desktop with Flatpak, and a GPU with 8 GB or more of memory for training (24 GB for very
large scenes). NVIDIA cards (GTX 10-series to RTX 50-series) train with gsplat. AMD and Intel cards, integrated ones
included, train with the Vulkan trainer, which is experimental: it has not yet been compared with gsplat on real
GPUs. Without a usable GPU it can still view, edit and process datasets.

## Install

Install from the project's Flatpak repository, so updates arrive like any other Flatpak app:

```bash
flatpak install --user https://benevision.github.io/SplatWork/flatpak/splatworker.flatpakref
```

Or build it from this repository (needs `flatpak-builder`; it pulls `org.gnome.Platform//51` from Flathub). Copy
the project to a normal folder in your home directory first: USB and external drives are usually mounted
"noexec", where the build can't run.

```bash
cd linux/flatpak
bash build.sh         # builds and installs for your user
bash make-bundle.sh   # optional: a single-file dist/SplatWorker-<version>-x86_64.flatpak to share
```

A shared single-file bundle installs with `flatpak install --user SplatWorker-<version>-x86_64.flatpak`.

## First launch

Open **SplatWork for Linux** from the app grid. On first launch the setup assistant runs:

1. It finds your GPUs: an NVIDIA GPU and driver through NVML / `nvidia-smi` (with the highest CUDA version
   the driver supports), and AMD and Intel cards from the system's PCI devices and their Vulkan driver.
2. It lists what it will download and why, with sizes. That is about 2.7 GB for an NVIDIA setup, and about
   0.3 GB for the processing-only setup.
3. It installs a private Python 3.10, then the PyTorch wheel for your GPU (CUDA for NVIDIA, ROCm for a
   dedicated AMD card, XPU for an Intel Arc card, the CPU build otherwise), then gsplat with prebuilt kernels
   (NVIDIA only), the worker's libraries and splatprep (optional). Everything goes under
   `~/.var/app/com.example.SplatWorker/data/splatworker/`. You get a real progress bar, plus Cancel and
   Retry. Finished downloads are cached, so Retry continues from the step that failed.
4. It checks that PyTorch can see the GPU, then offers to start the server.

On an AMD or Intel card, the assistant says that the card trains with the Vulkan trainer (experimental). That
trainer needs no download; the processing runtime adds dataset preparation, on the GPU through ROCm or XPU where
it can and on the CPU otherwise. Without any usable GPU, the assistant explains that training is not possible on
the CPU and offers the processing-only runtime, so the Mac can still send dataset processing jobs (§6).

After setup, the **Queue & Worker** page holds the worker side of the app:

* **Status**: the server switch, the address(es) and port, the Bonjour name, and the access token
  (copy, regenerate, optional QR code). It also shows what the worker can do, a live GPU card (memory
  used/free, utilisation, temperature, power), the Macs connected in the last 5 minutes, and recent
  requests.
* **Jobs**: progress bars with step/total, PSNR, splat count and time left. Each job has cancel,
  delete, log, and "save result" buttons (saves to Downloads).
* **Log**: the server's output.
* **Connect**: step-by-step instructions for SplatWork on the Mac, firewall commands and checks.
* **Settings** (Ctrl+,): port, Bonjour name, token on/off, data folder, maximum upload size, start the
  server when the app opens, keep running when the window is closed, start at login (through the XDG
  Background portal), install updates automatically, log detail, and repair/remove the GPU runtime. Each
  setting is explained right under it. Also:
  * **GPUs**: *Use all GPUs*, or tick the GPUs to use. Each ticked GPU trains one job at a time. An
    integrated GPU beside a faster card starts unticked. *Measure GPU speed* trains the benchmark scene on
    each GPU for about a minute with the Vulkan trainer (`splatworker measure-gpus` does the same); a GPU
    under a quarter of the fastest counts as slower. Each slower GPU has a choice: *Only when faster GPUs
    are busy* (default) or *Always use, like any other GPU*. *Move jobs to a faster GPU when one is free*
    lets a job a slower GPU started pause and continue on the faster GPU from the same step (Vulkan trainer
    on both).
  * **Training**: *Trainer* (Automatic, gsplat (NVIDIA), Vulkan (experimental); Automatic uses gsplat on
    NVIDIA and the Vulkan trainer on AMD and Intel), *Decode video on* (Automatic uses the NVIDIA GPU's
    video decoder, NVDEC, when it gives the same pictures as the CPU; or CPU), and *Processing jobs at once*
    (1 by default: processing runs on the CPU and takes the GPU only for its GPU steps).
  * **Startup and background → Check for updates**: *Check Now*, also in the main menu as **Check for
    Updates…**.
* **Models & Benchmarks → Compare trainers** trains the benchmark scene on the same GPU with gsplat and then
  with the Vulkan trainer, and shows speed, peak GPU memory, splats, held-out PSNR and SSIM side by side
  (`splatworker benchmark --compare-trainers`).

### Headless / without the window

The Flatpak also contains the plain server. It uses the settings and token from the app:

```bash
flatpak run com.example.SplatWorker --headless                       # = splatworker serve with the app's settings
flatpak run com.example.SplatWorker --headless --port 9000 --no-mdns  # any `serve` flag overrides them
flatpak run com.example.SplatWorker --headless info                   # other sub-commands run in the GPU runtime
flatpak run com.example.SplatWorker --install-runtime --yes           # setup assistant in a terminal (SSH boxes)
flatpak run --command=splatworker com.example.SplatWorker info        # same as `splatworker info`
cat ~/.var/app/com.example.SplatWorker/config/splatworker/token       # the token for the Mac
```

For a systemd **user** service, put `ExecStart=/usr/bin/flatpak run com.example.SplatWorker --headless` in
`~/.config/systemd/user/splatworker.service`. Then run `loginctl enable-linger $USER` so it runs without a
login. The native `splatworker serve` / `install.sh` / Docker paths below are unchanged.

### GPU runtimes and trainers

The setup assistant picks the runtime from your GPU and driver:

| GPU and driver | Runtime | Trains with |
|---|---|---|
| NVIDIA, driver 570+ (CUDA 12.8): RTX 20-series → RTX 50-series (compute 7.5–12.0) | `cu128`: PyTorch 2.7.1 + **gsplat 1.6 development build**, compiled by this repository's GitHub Actions from the commit in `flatpak/gsplat-pin.env` | gsplat |
| NVIDIA, driver 520–569: GTX 10-series and newer (not RTX 50) | `cu124` / `cu121` / `cu118`: PyTorch 2.4.1 + gsplat 1.5.3 (official prebuilt wheels) | gsplat |
| AMD, dedicated card | `rocm6.4`: PyTorch 2.8 for ROCm 6.4, processing only | Vulkan trainer |
| Intel Arc | `xpu`: PyTorch 2.7.1 for Intel GPUs, processing only | Vulkan trainer |
| Integrated GPU, or no GPU | CPU build, processing only | Vulkan trainer (integrated GPU), or none |

The Vulkan trainer (`splatwork-vk-train`, from `vktrain/`) is built into the Flatpak and needs no download. It
reads the same training settings as gsplat. On NVIDIA it can be chosen in *Settings → Training → Trainer*. It is
experimental: so far it has been checked against gsplat only on Mesa's software Vulkan driver. Auto clean still
needs gsplat. More in [docs/dev/GPU_SUPPORT.md](../docs/dev/GPU_SUPPORT.md).

gsplat 1.6 isn't released yet, so the pinned development commit is compiled once per pin (CI caches it) and
published next to the Flatpak at `…/wheels/`. To try a newer gsplat, change `GSPLAT_COMMIT` in
`flatpak/gsplat-pin.env` and push. Existing installs switch runtime when you run **Set Up…** again.

### How the Flatpak gets CUDA

* **Driver libraries** (`libcuda.so.1`, `libnvidia-ml.so.1`) come from the
  `org.freedesktop.Platform.GL.nvidia-<driver version>` extension. Flatpak installs it automatically for
  the host's driver on `flatpak update`. It must match the kernel driver *exactly*.
* **CUDA runtime**: it comes inside PyTorch's pip wheels (the `nvidia-*-cu12` packages). No CUDA
  toolkit is needed.
* **Why a separate Python 3.10**: gsplat's prebuilt CUDA wheels exist only for CPython 3.10 with
  PyTorch ≤ 2.4 and CUDA 11.8 / 12.1 / 12.4. The GNOME runtime's Python is 3.14 and the sandbox has no
  `nvcc` or C++ compiler for gsplat's JIT build. So the assistant downloads a SHA-256-pinned
  [python-build-standalone](https://github.com/astral-sh/python-build-standalone) CPython 3.10 and runs
  the worker with it as a child process of the GUI. The worker code is read straight from the app
  (`PYTHONPATH`), so app updates don't need a runtime reinstall.
* **Driver → wheel**: it picks the newest build that is not newer than the driver's CUDA version: ≥ 12.4
  → `cu124`, 12.1–12.3 → `cu121`, 11.8–12.0 → `cu118`. You can override it in the assistant's
  "Advanced" section.
* **RTX 50-series (Blackwell)**: PyTorch 2.4 has no kernels for them, so they need the `cu128` runtime
  (driver 570 or newer). With an older driver the assistant asks you to update it.

### Network behaviour

`--share=network` shares the host's network namespace, so nothing is NATed or proxied. The server binds
TCP 7878 (or your port) on the host's interfaces. python-zeroconf sends and answers mDNS multicast on
UDP 5353 itself, next to `avahi-daemon` (`SO_REUSEPORT`). The manifest also allows talking to
`org.freedesktop.Avahi`, but the worker does not use it today. Host firewall rules apply exactly as for
a native program. Discovery has the same limits as described in
[How the Mac finds the worker](#how-the-mac-finds-the-worker). Filesystem access is limited to the
app's own data folder and `~/Downloads` (result export). A different data folder chosen in Settings is
shared through the file chooser portal.

### Flatpak troubleshooting

* **Setup says "NVIDIA driver libraries are missing in the sandbox"**, or **"CUDA is not available"**
  after a driver update: the Flatpak NVIDIA extension does not match the host driver. Run
  `flatpak update`, then restart the app. If that doesn't add it (it doesn't always, for example when
  the flathub remote is in a different installation from the app), install it by name, with the driver
  version's dots turned into dashes: `flatpak install flathub org.freedesktop.Platform.GL.nvidia-620-70-06//1.4`.
  The setup screen shows the exact command for your driver. Compare `cat /proc/driver/nvidia/version` with
  `flatpak list --runtime | grep -i nvidia`; the versions must be identical
  (e.g. `550.54.14` ↔ `GL.nvidia-550-54-14`). Reboot if you updated the driver without rebooting.
* **The Mac doesn't find the worker**: open **TCP 7878** (or your port) and **UDP 5353** (mDNS) to
  your LAN. For ufw: `sudo ufw allow from 192.168.0.0/16 to any port 7878 proto tcp` and
  `... port 5353 proto udp`. For firewalld:
  `sudo firewall-cmd --permanent --add-port=7878/tcp --add-service=mdns && sudo firewall-cmd --reload`.
  Otherwise, add the worker on the Mac by the address shown on the Status page.
* **"Port 7878 is already in use"**: another worker is running, such as the systemd service or the
  Docker container. Stop it, or choose another port in Settings.
* **Setup fails with a network or disk error**: press Retry. Downloads resume from the failed step. The
  full log is in `~/.var/app/com.example.SplatWorker/data/splatworker/install.log`. The runtime needs
  about 6 GB of free space.
* **`flatpak update` says "Can't pull from untrusted non-gpg verified remote"**: a system-wide install made
  from the old unsigned ref. Reinstall for your user, as in [Updates from GitHub](#updates-from-github-recommended).
* **Start over**: Settings → GPU runtime → Remove, then Open Setup. To wipe everything, including
  jobs and settings: `flatpak uninstall --delete-data com.example.SplatWorker`.

### Updates from GitHub (recommended)

Every push to `main` that touches `linux/` or `tools/` is built by GitHub Actions
(`.github/workflows/splat-worker-flatpak.yml`) and published as a Flatpak repository on GitHub Pages.
Install from there once and updates arrive like any other Flatpak app: GNOME Software / KDE Discover
offer them, or run `flatpak update`. From 1.3.16 the app also updates itself: it installs a newly published
version through the Flatpak portal and restarts into it once no job is running or queued (Settings →
Startup and background → Install updates automatically; off = a banner with Update / Restart buttons).
**Check for Updates…** (main menu, or *Check Now* in the same Settings group) checks at once; *Update and
Restart* installs a newer version straight away, and with jobs running or queued it offers *Update After
Jobs* or *Stop Jobs and Update*. Headless installs (`--headless`, systemd) don't self-update: use
`flatpak update` (e.g. from a timer).

```bash
# one time: remove a locally built copy, then install from the published repository
flatpak uninstall --user com.example.SplatWorker
flatpak install --user https://benevision.github.io/SplatWork/flatpak/splatworker.flatpakref
```

**Signed repository.** From 2026-11-16 the repository is signed with GPG, and the `.flatpakref` carries the
public key, so Flatpak checks every update.

* Installs made earlier from the unsigned ref keep updating as before: their remote has signature checking
  turned off. To turn it on, uninstall and install again from the ref (the commands above).
* A **system-wide** install made from the old unsigned ref fails `flatpak update` with "Can't pull from
  untrusted non-gpg verified remote". Remove it and install for your user:
  ```bash
  flatpak uninstall --system com.example.SplatWorker
  flatpak install --user https://benevision.github.io/SplatWork/flatpak/splatworker.flatpakref
  ```
* A fork that publishes without a signing key (no `FLATPAK_GPG_PRIVATE_KEY` secret) gets an unsigned
  repository. Add its remote with `flatpak remote-add --user --no-gpg-verify splatworker <fork>/flatpak/repo/`,
  then `flatpak install --user splatworker com.example.SplatWorker`.

Settings, the token and the downloaded GPU runtime live in `~/.var/app/com.example.SplatWorker` and are
kept across updates and reinstalls, including the reinstalls above. Repository setup (one time): GitHub → Settings → Pages → Source:
"GitHub Actions"; GitHub Pages needs a public repository on free plans.

### Building and updating the Flatpak

* `flatpak/com.example.SplatWorker.yml`: manifest (org.gnome.Platform//51). Modules: bundled Python
  deps, splatprep source (`../tools`), the worker + GUI.
* `flatpak/python3-deps.json` is generated from `flatpak/requirements-flatpak.txt` by
  `./generate-deps.sh` (pip resolver, cp314 wheels, URLs + sha256). `./generate-deps.sh --pip-generator`
  uses flatpak-pip-generator instead. Re-run it when you bump `runtime-version` (Python version).
* GPU runtime pins (Python build, torch/gsplat per CUDA tag) live in `splatworker/gui/runtime_plan.py`.

## Requirements

* Linux with an NVIDIA GPU, a recent driver (`nvidia-smi` works), and 8 GB+ VRAM. Use 24 GB for large
  scenes with `maxGaussians` set to 3M or more. The native install sets up gsplat (CUDA) only. To train on
  AMD or Intel without the Flatpak, build the Vulkan trainer in `vktrain/` with CMake and put
  `splatwork-vk-train` on `PATH` (or point `SPLATWORK_VK_TRAIN` at it).
* Python 3.10+.
* The CUDA toolkit (`nvcc`). You need it only if no prebuilt gsplat wheel matches your torch/CUDA
  version: gsplat then compiles its kernels once on first use. fused-ssim always needs it.

## Install (native)

```bash
cd linux
./install.sh                    # venv in ./venv, picks the PyTorch CUDA build from your driver
./install.sh --cuda cu121       # or force a wheel tag (cu118 cu121 cu124 cu126 cu128)
./install.sh --with-splatprep   # also install ../tools/splatprep for processing jobs (§6)
./install.sh --system           # + 'splatworker' user, /etc/splatworker/token, systemd service
```

Manual install:

```bash
python3 -m venv venv && . venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install "gsplat>=1.4,<2"            # or a prebuilt wheel: https://docs.gsplat.studio/whl/
pip install -e ".[train]"
pip install --no-build-isolation git+https://github.com/rahul-goel/fused-ssim   # optional
pip install -e ../tools/splatprep                                                # optional (§6)
```

Check what the worker sees:

```bash
splatworker info
```

## Run

```bash
splatworker gen-token --out ~/.config/splatworker/token    # prints the token and saves it (mode 600)
splatworker serve --port 7878 --data-dir ~/splatworker-data \
                  --token-file ~/.config/splatworker/token --name "Render Box"
```

Options: `--gpus 0,1` runs one job per listed GPU at a time. It takes CUDA indices and `vk:<PCI address>`
keys for GPUs only the Vulkan trainer can use (AMD, Intel, integrated), e.g. `--gpus 0,vk:0000:c5:00.0`;
`splatworker info` lists the keys. The default is every GPU except an integrated one beside a faster card.
`--trainer auto|gsplat|vulkan` picks the trainer (default `auto`). `--gpu-mode vk:0000:11:00.0=always` uses a
slower GPU like any other (default `auto`: only while faster GPUs are busy), and `--no-migrate` keeps jobs where
they started. `--video-decode auto|cpu` sets where processing decodes video, and `--cpu-jobs N` how many
processing jobs run at once (default 1).
`--no-mdns` turns off Bonjour, and `--advertise-ip 192.168.1.20` sets the advertised address.
`--max-upload-gb` and `--max-extracted-gb` cap upload sizes (defaults 20 and 40). Most flags also have an
environment variable: `SPLATWORKER_TOKEN`, `SPLATWORKER_TOKEN_FILE`, `SPLATWORKER_DATA_DIR`,
`SPLATWORKER_PORT`, `SPLATWORKER_NAME`, `SPLATWORKER_GPUS`, `SPLATWORKER_TRAINER`, `SPLATWORKER_VIDEO_DECODE`,
`SPLATWORKER_CPU_JOBS`.

### As a service

```bash
./install.sh --prefix /opt/splatworker/venv --system
journalctl -u splatworker -f
```

`splatworker.service` runs as the unprivileged `splatworker` user, with data in `/var/lib/splatworker`
and the token in `/etc/splatworker/token`.

### Docker

```bash
docker build -t splatworker .
docker run -d --gpus all --network host --restart unless-stopped \
  -v splatworker-data:/data -e SPLATWORKER_TOKEN="$(cat token)" splatworker
# with splatprep (BuildKit named context):
docker build --build-context tools=../tools --target worker-splatprep -t splatworker .
```

Docker needs the NVIDIA Container Toolkit. `--network host` is required for mDNS to reach the LAN. With
port mapping instead (`-p 7878:7878`), discovery fails and you have to add the worker by address in
SplatWork.

### Local training without the server

```bash
splatworker train path/to/dataset --out scene.ply --set strategy=adaptive --set iterations=7000 --preview
splatworker train path/to/dataset --out scene.ply --config config.json      # TrainingConfig JSON
```

## How the Mac finds the worker

The worker advertises the Bonjour service `_splatworker._tcp` on the configured port with these TXT
records:

* `version=1`
* `gpu=<GPU name>`, or `2x <name>` when there are several GPUs
* `auth=1|0`

SplatWork lists it under remote workers. When `auth=1` it asks for the token and then sends
`Authorization: Bearer <token>` on every request. To check from the Mac:

```bash
dns-sd -B _splatworker._tcp            # browse
dns-sd -L "Render Box" _splatworker._tcp   # resolve host/port/TXT
curl -H "Authorization: Bearer $TOKEN" http://render-box.local:7878/v1/info
```

mDNS does not cross subnets or VLANs, and some Wi-Fi networks block multicast (guest networks, "client
isolation"). In those cases, add the worker in SplatWork by IP and port.

## API summary

Every field is described in [docs/PROTOCOL.md](../docs/PROTOCOL.md); this is a summary. All endpoints need
`Authorization: Bearer <token>` when a token is configured; without it they return
`401 {"error": ...}`. All errors are `{"error": "message"}`.

| Method & path | Notes |
|---|---|
| `GET /v1/info` | name, version, workerVersion, gpus, backend, busy, queueLength, capabilities (`train` if a trainer is available, `process` if splatprep imports), models. Newer workers also report `adapters` (every graphics card), `trainer` (which trainer trains), `trainers` (the ones a job may ask for), `slots` (one per GPU, with measured speed, `slower`, `userChoice`), `vulkanDevices` (every Vulkan GPU's measured speed) and `cpuJobs` (the processing pool) |
| `POST /v1/datasets` | raw zip. A dataset root must contain `dataset.json` (training) or `images/` (processing). It can sit at the zip root or inside one top-level folder. Returns `{"datasetId"}` |
| `GET /v1/datasets`, `DELETE /v1/datasets/{id}` | extras for disk management (`409` while a job uses it) |
| `POST /v1/jobs` | `{"datasetId","name","config":{TrainingConfig}}` returns `{"jobId"}`. Optional `"slot"` (a `/v1/info` slot id) and `"trainer"` (`gsplat` or `vulkan`) |
| `POST /v1/process` | `{"datasetId","request":{ProcessingRequest}}` returns `{"jobId"}` (§6) |
| `GET /v1/jobs`, `GET /v1/jobs/{id}` | JobStatus. `kind` is `train` or `process`. Extra fields include `datasetId`, `createdAt`, `gpu`, `slot`, `trainer`, `evalSsim` and `resource` (`cpu`, `gpu` or `waiting-gpu` for processing jobs) |
| `POST /v1/jobs/{id}/cancel` | A queued job is cancelled at once. A running job keeps its state with `message: "cancelling"` until the trainer stops, then becomes `cancelled` |
| `DELETE /v1/jobs/{id}` | `204`; cancels the job if it is running, then deletes its files |
| `GET /v1/jobs/{id}/preview.ply` | latest snapshot (`404` before the first one), header `X-Preview-Step` |
| `GET /v1/jobs/{id}/result.ply` | `404` until `state == "completed"` |
| `GET /v1/jobs/{id}/output.zip` | processing output (`outputDir` zipped, contents at the root) |
| `GET /v1/jobs/{id}/log` | text/plain log |

Job states: `queued`, then `loading`, then `training`, and finally `completed`, `failed` or `cancelled`.
Processing jobs go from `loading` straight to a final state. For them, `step`/`totalSteps` is progress
×1000 and `message` is the current stage.

Jobs persist in `<data-dir>/jobs/<id>/job.json`. When the worker restarts, it re-queues jobs that were
`queued` and marks jobs that were running as `failed` ("interrupted by worker restart").

## Training details (TrainingConfig → gsplat)

| Config | Implementation |
|---|---|
| `strategy: "mcmc"` | `gsplat.strategy.MCMCStrategy(cap_max=maxGaussians, noise_lr=noiseLR, refine_start_iter=densifyStart, refine_stop_iter=densifyStop or 25000, refine_every=densifyInterval, min_opacity=pruneOpacity)` |
| `strategy: "adaptive"` | `DefaultStrategy(prune_opa, grow_grad2d=densifyGradThreshold (×4 with absGrad), reset_every=opacityResetInterval, absgrad=absGrad, …)`. Growth pauses once `maxGaussians` is reached; pruning continues |
| `shDegree`, `shDegreeInterval` | active degree = `min(step // interval, shDegree)` |
| `lrPosition` → `lrPositionFinal` | Adam LR × `sceneExtent`, exponential decay over `iterations` |
| `ssimWeight` | `(1-λ)·L1 + λ·(1-SSIM)`, using fused-ssim if installed, else a torch SSIM (11×11 Gaussian, valid padding) |
| `opacityReg`, `scaleReg` | L1 on `sigmoid(opacity)` and `exp(scale)`; default 0 for ADC unless set |
| `antialiased` | `rasterize_mode="antialiased"` (otherwise `"classic"`) |
| `depthWeight` | `render_mode="RGB+ED"`; L1 between expected depth and the depth map on pixels where depth > 0. The depth map is nearest-resized to the training resolution. Defaults to 0.1 when the train cameras have depth |
| `background` | composited as `rgb + (1-alpha)·bg`, with a new colour each step for `random`. Images with an alpha channel have their ground truth composited with the same bg |
| `maxImageSize` | images are downscaled with Lanczos so the longest side fits, and intrinsics are scaled to match |
| `evalInterval` | PSNR on `holdout` cameras. Without holdout cameras, `psnr` is a smoothed train-batch PSNR |
| `previewInterval` | `preview.ply` is written atomically every N steps |
| `region` (§7) | Gaussians outside the box expanded by `margin × size` per axis get their opacity set to about 0 at every densify step, so ADC prunes them and MCMC relocates them. Initial points are filtered to the box (random init fills the box). Only cameras whose frustum sees the box are used. The exported PLY is cropped to the expanded box |
| `cameraIds` | restricts the cameras used (train and holdout) |

Initialisation uses `points3d.ply` (scales are set from the mean distance to the 3 nearest neighbours,
using scipy if available). Without it, `randomInitCount` points are placed uniformly in a cube of
half-size 1.5 × `sceneExtent` around `sceneCenter`.

The output PLY is the INRIA 3DGS layout: `x y z nx ny nz f_dc_0..2 f_rest_0..44 opacity scale_0..2
rot_0..3`. `f_rest` is channel-major (as in the original 3DGS export), opacity is a logit, scale is a
log value, and the rotation is a normalised `w,x,y,z` quaternion. Lower SH degrees are zero-padded.

Each job runs in its own spawned Python process pinned to one GPU (with the Vulkan trainer, a
`splatwork-vk-train` process). GPU memory goes back to the driver after every job, and a crash in the trainer
doesn't take the API down. The Vulkan trainer reads the same TrainingConfig; how it maps the settings is in
[docs/dev/GPU_SUPPORT.md](../docs/dev/GPU_SUPPORT.md).

## Firewall

Open TCP 7878, plus UDP 5353 for mDNS, to your LAN only:

```bash
# ufw
sudo ufw allow from 192.168.1.0/24 to any port 7878 proto tcp
sudo ufw allow from 192.168.1.0/24 to any port 5353 proto udp
# firewalld
sudo firewall-cmd --permanent --add-port=7878/tcp --add-service=mdns && sudo firewall-cmd --reload
```

## Security notes

* **Always set a token.** Without one, anyone who can reach the port can upload data and run GPU
  jobs. Generate it with `splatworker gen-token`. Pass it with `--token-file` or `SPLATWORKER_TOKEN`
  rather than `--token`, because command lines are visible to other local users through `ps`. The
  server compares tokens in constant time.
* **LAN only.** The API is plain HTTP, so the token and your data are visible on the network. Don't
  port-forward it to the internet. For remote use, put it behind a VPN (WireGuard, Tailscale) or a TLS
  reverse proxy (Caddy or nginx), and use `--host 127.0.0.1` when proxying.
* **Uploads are treated as untrusted:**
  * Zips are rejected if they contain absolute paths, `..`, backslashes, symlinks, special files or
    encrypted entries.
  * Size is capped for the upload, the total extracted data (counted during extraction) and the file
    count. A per-entry compression-ratio limit catches zip bombs.
  * Paths inside `dataset.json` and processing requests have to stay inside the dataset.
  * Job and dataset IDs are validated against `[A-Za-z0-9_-]`.
* The data dir is created with mode 700. The systemd unit runs unprivileged with
  `ProtectSystem=strict`.
* Outputs from processing jobs are zipped without following symlinks that point outside the job or
  dataset directories.

## Troubleshooting

* **`capabilities` is missing `train` / jobs fail with "PyTorch is not installed" or "gsplat is not
  installed"**: run the worker from the venv that install.sh created. Use `splatworker info` to check.
* **"CUDA is not available to PyTorch"**: the torch build doesn't match the driver. Reinstall with
  `--cuda` set to a tag no newer than the `CUDA Version` that `nvidia-smi` reports. In Docker, pass
  `--gpus all`.
* **The first job sits in `loading` for minutes**: gsplat is JIT-compiling its CUDA kernels. This
  happens once, and the result is cached in `TORCH_EXTENSIONS_DIR` (or `~/.cache/torch_extensions`).
  It needs `nvcc` and `ninja`. Installing a prebuilt wheel avoids it.
* **`CUDA out of memory`**: lower `maxGaussians` and/or `maxImageSize`. MCMC never goes over its cap.
  ADC can overshoot by up to one densify pass.
* **The Mac doesn't see the worker**:
  * Check UDP 5353 in the firewall.
  * Make sure both machines are on the same subnet and the Wi-Fi allows multicast.
  * Check `avahi-daemon`. It can coexist with the worker, but some hardened setups block other mDNS
    responders.
  * In Docker, use `--network host`.
  * Or add the worker by IP.
* **Wrong address advertised** (Docker bridge, VPN): use `--advertise-ip <LAN IP>`.
* **401 errors**: the token in SplatWork doesn't match `/etc/splatworker/token`. Whitespace and
  newlines are stripped from token files.
* **Processing jobs fail with "splatprep is not installed"**: run `./install.sh --with-splatprep`, or
  `pip install -e ../tools/splatprep` inside the venv. `models` in `/v1/info` lists what imports.
* **Disk fills up**: uploaded datasets are kept until you `DELETE /v1/datasets/{id}` or remove
  `<data-dir>/datasets/<id>`. Deleting a job removes its files.
* Logs are available from `GET /v1/jobs/{id}/log`, `<data-dir>/jobs/<id>/train.log`, and
  `journalctl -u splatworker`.

## Development / tests

The server, dataset parsing and PLY IO import torch and gsplat lazily, so the tests run without a GPU:

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install fastapi uvicorn httpx pytest zeroconf numpy pillow pyyaml
pytest -q
```

The desktop app's GTK-free logic is tested the same way (`tests/test_gui_*.py`). That covers the
setup planner (driver → wheel mapping), the pip progress parser, the installer (with a fake Python
archive and fake pip), the NVML wrapper (with a fake `pynvml`), settings and token storage, and the
server lifecycle controller, which starts the real `python -m splatworker serve`.
`tests/test_flatpak_files.py` validates the manifest, `python3-deps.json`, the desktop file, metainfo
and icons, and byte-compiles the GTK files. Run the GUI from a checkout on Linux with
`python3 -m splatworker.gui.launcher` (needs `python3-gi`, GTK 4 and libadwaita ≥ 1.4).

The tests use a fake in-thread backend and a stand-in `splatprep` package. One test runs the real
spawn-based subprocess backend for a processing job, and another checks that the training subprocess
fails cleanly when torch is missing.

### Building gsplat 1.6 on your own PC

The "PyTorch 2.7 + CUDA 12.8 · gsplat 1.6" runtime needs a gsplat wheel compiled for the pinned development
commit (`flatpak/gsplat-pin.env`). Instead of waiting for GitHub Actions (hours on 4 vCPUs), build it on the
worker PC in a throwaway container:

```bash
sudo pacman -S podman          # CachyOS / Arch (Fedora: sudo dnf install podman)
cd linux/flatpak
./build-gsplat-wheel.sh        # just this PC's GPU (e.g. RTX 4080 → compute capability 8.9) — minutes
./build-gsplat-wheel.sh --all  # every supported card (RTX 20 – RTX 50) — for publishing
```

The wheel is copied to SplatWork for Linux's `wheels/` folder (`~/.var/app/com.example.SplatWorker/data/splatworker/wheels`)
and Set Up uses it from there. To publish an `--all` build for everyone, create a GitHub Release tagged
`gsplat-1.6.0-cu128` and attach the `.whl`: the Flatpak workflow then uses it instead of compiling. When gsplat 1.6
is officially released with prebuilt wheels, switch the profile back to the public index and delete the release.
