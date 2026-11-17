# Testing SplatWork for Linux on an AMD GPU under SteamOS

For the RX 6700 XT test PC (RDNA 2, gfx1031, 12 GB) running SteamOS. Everything comes through the Flatpak: SteamOS's
system is read-only and nothing is installed on it. PyTorch for ROCm brings AMD's GPU libraries with it; the host
only needs the amdgpu driver (SteamOS has it) and `/dev/kfd` + `/dev/dri`, which the app's `--device=all`
permission gives it.

Do this in **Desktop Mode**, in **Konsole**, as the normal user (`deck`). Each step prints something to paste back;
the important lines start with `SPLATWORK-`. The app sets `HSA_OVERRIDE_GFX_VERSION=10.3.0` for this card itself
(inside the app only), so don't set it in the shell.

## 1. Install

```bash
flatpak install --user https://benevision.github.io/SplatWork/flatpak/splatworker.flatpakref
```

Already installed? Update instead: `flatpak update --user com.example.SplatWorker`. Check the version (1.7.0 or
newer):

```bash
flatpak info --user com.example.SplatWorker | grep -i version
```

## 2. Detection (no runtime needed yet)

```bash
ls -l /dev/kfd /dev/dri; flatpak run --command=sh com.example.SplatWorker -c 'ls -l /dev/kfd /dev/dri'
```

```bash
flatpak run com.example.SplatWorker --cli info
```

Paste both outputs. Expected: `/dev/kfd` exists inside and outside the sandbox; `"adapters"` lists the card as
`"vendor": "amd"`, about 12288 MB, `"gfx": "gfx1031"`.

Then open **SplatWork** from the application menu once: the setup should say *"AMD Radeon RX 6700 … found"* and
offer *"Processing runtime … ROCm"* with a note about gfx1031 / 10.3.0. A screenshot of that page helps. Close it
with **Not Now** (the next step installs from the terminal, which keeps a log).

## 3. PyTorch for ROCm (processing runtime) and the GPU check

About 5 GB download, 14 GB disk:

```bash
flatpak run com.example.SplatWorker --install-runtime --profile rocm6.4 --lightglue --yes 2>&1 | tee ~/splatwork-install.log | tail -40
```

```bash
flatpak run com.example.SplatWorker --cli gpu-check --no-train 2>&1 | tee ~/splatwork-gpucheck.txt
```

Paste `~/splatwork-gpucheck.txt` (at least the last line, `SPLATWORK-GPU-CHECK {...}`). What we look for: torch
`hip` set, the device named with `gfx1031` or `gfx1030`, `alloc_1gb`, `matmul_fp32` (TFLOPS, `relError` below 0.01) and
`conv_backward` all ✓, splatprep device `cuda` with backend `rocm`.

If PyTorch can't see the card, also paste:

```bash
flatpak run --command=sh com.example.SplatWorker -c 'env | grep -E "HSA|HIP|ROC"; ls /sys/class/kfd/kfd/topology/nodes/*/'
```

### Steam Deck and other integrated AMD GPUs: why processing runs on the CPU (1.7.13)

On a Steam Deck (Van Gogh APU, `gfx1033`, "AMD Custom GPU 0405") the setup assistant installs the **CPU** build of
PyTorch on purpose: `gui/runtime_plan.processing_profile_for` gives integrated GPUs no ROCm profile, because ROCm
doesn't support APUs and its memory comes from the small fixed VRAM carve-out (1 GB on the Deck unless *UMA Frame
Buffer Size* is raised to 4 GB in its BIOS). So `/v1/info` shows `"processing": {"device": "cpu", "reason": "PyTorch
can't use the AMD Custom GPU 0405"}`. The ROCm profile was never installed, so the `gfx1033 → 10.3.0` override
(`gpu_vendors.HSA_OVERRIDES`, set by `gui/installer.py` inside the app only) never applied. `/dev/kfd` is visible in
the Flatpak (`--device=all`) when SteamOS's kernel exposes it.

Since 1.7.13 a camera solve without a GPU for PyTorch uses **COLMAP SIFT** (OpenCV SIFT if pycolmap's isn't there) instead of SuperPoint + LightGlue, unless
the feature model was chosen explicitly. The job log, the prep screen and the setup page say so.

To try ROCm on the Deck anyway (experimental; it may fail or run out of memory), raise the UMA frame buffer to 4 GB,
then:

```bash
flatpak run com.example.SplatWorker --install-runtime --profile rocm6.4 --lightglue --yes 2>&1 | tee ~/splatwork-install.log | tail -40
```

```bash
flatpak run com.example.SplatWorker --cli gpu-check --no-train 2>&1 | tee ~/splatwork-gpucheck.txt
```

```bash
flatpak run --command=sh com.example.SplatWorker -c 'ls -l /dev/kfd; env | grep -E "HSA|HIP|ROC"; ls /sys/class/kfd/kfd/topology/nodes/*/'
```

Send back the last line of the GPU check (`SPLATWORK-GPU-CHECK {...}`): torch `hip` set, a device named with
`gfx1030` (the override) and `alloc_1gb` / `matmul_fp32` passing mean ROCm works there. To go back to the CPU
build: *Set Up GPU Runtime… → Advanced → CPU*.

### GPU matching benchmark (1.7.14, docs/dev/GPU_PROCESSING.md stage 1)

In Konsole on the Deck, after it has updated to SplatWork for Linux 1.7.14 (`flatpak update --user
com.example.SplatWorker` if it hasn't yet). Nothing here needs the server's token or the network.

1. The synthetic benchmark (about a minute):

```bash
flatpak run --command=splatwork-vk-match com.example.SplatWorker bench | tail -3
```

2. Real photos: the images of one of your SplatWork projects (list them with `ls ~/Documents/SplatWork`), OpenCV on
   the CPU against the GPU (a few minutes):

```bash
flatpak run --command=sh com.example.SplatWorker -c '"$XDG_DATA_HOME"/splatworker/venv/bin/python /app/share/splatwork/match_bench.py run "$HOME/Documents/SplatWork/PROJECT/dataset/images" --images 30'
```

   (replace PROJECT with the project's folder name). Send back the last line of each: `{"type": "bench", …}` and
   `{"type": "match-bench", …}` — GPU and CPU milliseconds per pair, the speed-up and whether the matches were
   identical.

## 4. Training with the Vulkan trainer (experimental)

SplatWork 1.7.0 and newer have a built-in Vulkan trainer: it is part of the app (nothing to download) and
trains on this card by default. It uses SteamOS's own AMD Vulkan driver (RADV, from the Flatpak runtime). This is
its first run on a real GPU.

```bash
flatpak run com.example.SplatWorker --cli info 2>&1 | tee ~/splatwork-info.txt | grep -A9 -E '"trainer"|"vulkanTrainer"'
```

Expected: `"trainer"` → `"kind": "vulkan"`, and `"vulkanTrainer"` → `"ok": true`, a device like
`AMD Radeon RX 6700 XT (RADV NAVI22)`, `"deviceType": "discrete"`, about 12 GB. If `"ok"` is false, paste the
whole `~/splatwork-info.txt` and also:

```bash
flatpak run --command=splatwork-vk-train com.example.SplatWorker --info; flatpak run --command=sh com.example.SplatWorker -c 'ls /usr/share/vulkan/icd.d/ /usr/lib/x86_64-linux-gnu/GL/*/vulkan/icd.d/ 2>&1'
```

Then the standard benchmark (the same one the Mac runs; about 10–30 minutes). Open SplatWork, make sure the server
is **on** (Queue & Worker page), then:

```bash
flatpak run com.example.SplatWorker --cli benchmark --quick 2>&1 | tail -15
```

```bash
flatpak run com.example.SplatWorker --cli benchmark 2>&1 | tee ~/splatwork-benchmark.txt | tail -30
```

The quick one (a tiny scene, 200 steps) only checks that training works. Paste the JSON at the end of the full one:
`trainStepsPerSecond`, `psnr` and `evalPsnr` are what count (the RTX 4080 PC's numbers are in its Models &
Benchmarks page). Also paste the job's log (Queue & Worker page, or the newest
`~/.var/app/com.example.SplatWorker/data/splatworker/data/jobs/*/train.log`): its first lines say which device and
which atomics path the trainer used, and its last ones give the time per step.

If training fails or the card hangs (a GPU reset shows as the screen flickering or `amdgpu: GPU reset` in
`journalctl -k`), paste the job log and `journalctl -k -b | grep -i amdgpu | tail -30`.

The experimental ROCm build of gsplat (runtime profile `rocm6.4-train`) is no longer needed: its build doesn't work
on Radeon cards (docs/dev/GPU_SUPPORT.md).

## 5. Optional: a real scene

Copy a small photo set (30–80 photos) to the PC, *New Project → Photos* in SplatWork, Balanced, and compare the result
with the same set trained on the RTX 4080 (PSNR on the Projects page, and how it looks).

## Undo

The runtime lives in the app's own folder; to remove everything: `flatpak uninstall --user --delete-data
com.example.SplatWorker`.

## What to send back

`~/splatwork-gpucheck.txt`, `~/splatwork-info.txt`, `~/splatwork-benchmark.txt` with the job log, the outputs of
step 2, and the install logs if a step failed.
