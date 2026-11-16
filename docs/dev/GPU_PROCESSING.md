# GPU processing without CUDA: feature extraction, matching, masks, depth

*Decision document, 2026-11-16. Research only, apart from the CPU fallback that 1.7.13 shipped (below). It covers splatprep's GPU work on the machines
where PyTorch gets no GPU: AMD APUs without ROCm (the Steam Deck's Van Gogh `gfx1033`, the Ryzen's Radeon 610M
`gfx1036`), older AMD cards, Intel iGPUs, and Windows on AMD or Intel. Companion to [GPU_SUPPORT.md](GPU_SUPPORT.md),
which covers training. The speed figures marked "est." are back-of-envelope numbers from peak FP32 rates, not
measurements. The benchmark plan below replaces them.*

## What 1.7.13 already does (measured)

Where PyTorch has no GPU, a camera solve whose features are "auto" or the untouched default now uses **COLMAP SIFT**
(pycolmap) on the CPU instead of SuperPoint + LightGlue (`splatprep/runner.cpu_features`; OpenCV SIFT if pycolmap's
SIFT is missing; an explicitly chosen model is kept, with a time warning). Measured on 4 CPU threads (Apple M-series,
roughly a Steam Deck's 4 Zen 2 cores), 184 frames of a room video at 1600 × 1200, sequential matching:

| CPU solver | Time | Photos placed | Mean reprojection error |
|---|---|---|---|
| **COLMAP SIFT** | **105 s** | **138 of 184** | 0.71 px |
| OpenCV SIFT + ratio test | 116 s (73 s of it matching, a Python loop) | 41 of 184 | 0.62 px |

So the CPU baseline for everything below is COLMAP SIFT, not OpenCV SIFT. SuperPoint + LightGlue couldn't be timed
here (no PyTorch on this Mac); the Deck's own runs took over 2 h before they were stopped. A Deck is slower than this
Mac per core, so expect several minutes, not seconds, for a few hundred photos even with SIFT.

## Decision for the user

**Recommendation: option 1, our own Vulkan compute for SIFT and matching**, built next to `splatwork-vk-train` as a
second small native program, `splatwork-vk-features`. It reuses `linux/vktrain`'s Vulkan layer, the device probe,
the build and the CI.

- **What it does:** GPU SIFT (DoG pyramid, orientation, descriptors) and brute-force mutual-nearest-neighbour
  matching with a ratio test, on every Vulkan 1.2 GPU: the Steam Deck, the 610M, older AMD (GCN 3 and later on
  RADV), Intel iGPUs (ANV, and the Windows drivers), and NVIDIA too. Lavapipe runs it in CI.
- **Why this option:** matching is most of the camera-solve time (pairs grow much faster than images), and SIFT plus
  brute-force matching is small and well understood. It needs **no processing-runtime download**, no PyTorch, no
  ROCm, and no new Python. The code is all ours (SIFT's patent expired in March 2020) and carries the same
  permissive licence as vktrain. It ships in the Flatpak and the Windows MSI the same way the trainer does.
- **What it does not do:** learned features (LightGlue), SAM 2 masks, sky masks and Depth Anything still run on
  the CPU on these machines. They are per-image costs, not per-pair costs, so they hurt less. A later stage can add
  ONNX Runtime's WebGPU execution provider for them (option 2, the runner-up). That is a separate decision.
- **Effort (est.):** about 5–7 weeks to a shipped stage 2: about 1 week for matching, 3–4 weeks for SIFT with
  checks against OpenCV, and 1–2 weeks for splatprep integration and the app.
- **Expected result on a Steam Deck (est.):** matching goes from about 0.5 s per pair (OpenCV on 4 Zen 2 cores) to
  about 5–15 ms per pair, and extraction goes from about 1 s per image to about 50–100 ms. The solve that took
  more than 2 h should then be limited by COLMAP/GLOMAP mapping on the CPU.

**What I need from you:** approval of stage 1 (the matching kernel plus the benchmark, about 1 week), with a
go/no-go after the Steam Deck numbers. If you would rather get learned features on these GPUs first, the
alternative is option 2 (ONNX Runtime + WebGPU). That needs the processing runtime moved from Python 3.10 to
3.11 or later, and the EP is beta.

## What splatprep needs, and what costs time

| Stage | Today | Cost scales with | GPU-portable today? |
|---|---|---|---|
| Features: SuperPoint / ALIKED / DISK, SIFT (`backends/lightglue_backend.py`, `opencv_sift.py`) | PyTorch on CUDA / ROCm / XPU / MPS, else COLMAP SIFT on the CPU (since 1.7.13; OpenCV SIFT if missing) | images | no |
| Matching: LightGlue, or OpenCV `BFMatcher.knnMatch` both ways plus a Python loop | the same | **pairs**: sequential with overlap 10 = 10·N; exhaustive up to N = 60 | no |
| Dense matchers (LoFTR, MASt3R, VGGT) | PyTorch | pairs or the whole set | no (CUDA-class GPUs only) |
| Mapping (COLMAP / GLOMAP via pycolmap) | CPU | images and tracks | n/a (CPU in every option) |
| SAM 2 masks, sky masks, Depth Anything V2 priors | PyTorch | images (SAM 2: video frames) | no |

At 300 photos with sequential matching, that is about 3,000 pairs against 300 extractions. At 0.5 s per pair on
the CPU, matching alone takes about 25 min. Our own Vulkan matcher removes most of that.

## Options compared

Speeds are per image (extraction) or per pair (matching) at about 4,096 keypoints and 1,600 px. "4080 LG" is the
CUDA baseline: SuperPoint + LightGlue on the RTX 4080, roughly 10 ms per image and 15–25 ms per pair (est., scaled
from LightGlue's published 50 FPS at 4,096 keypoints on an RTX 3080). The CPU baseline is COLMAP SIFT on the same machine
(measured above; OpenCV SIFT + `BFMatcher` places far fewer photos).

| Option | Licence | Maturity | Flatpak / Windows packaging | Covers | Speed on Deck / 610M (est.) | Effort |
|---|---|---|---|---|---|---|
| **1. Own Vulkan SIFT + BF matching (vktrain infra)** | ours; SIFT patent expired 2020 | new code; infra proven (kernels gradient-checked on lavapipe, ran on 4080 + 610M) | in the existing Flatpak module; the Windows build already planned for vktrain; no download | features (SIFT) + matching; LightGlue-style later | matching about 5 ms (Deck) / 15 ms (610M) per pair, 30–100× the CPU; about the 4080 LG per-pair time, with fewer good matches on hard pairs | medium (5–7 weeks) |
| 2a. ONNX Runtime **WebGPU EP** (Dawn → Vulkan / D3D12) | MIT; models: LightGlue Apache-2.0, ALIKED BSD-3, DISK Apache-2.0, **SuperPoint non-commercial**, Depth Anything V2 Small Apache-2.0 (Base/Large CC-BY-NC), SAM 2 Apache-2.0 | `onnxruntime-ep-webgpu` 0.5.0, "Beta", Oct 2026 | PyPI wheels for Linux x86-64 / aarch64 and Windows; **Python ≥ 3.11** (processing runtime is 3.10); needs `libvulkan.so.1` (in the Freedesktop runtime) | features + LightGlue + depth; SAM 2 partly (export is work) | unknown on RDNA 2 iGPUs: WebGPU has no float atomics and limited subgroups, and attention is memory-bound on a 1 GB carve-out. Likely 3–10× the CPU | small to integrate; risk sits in the EP |
| 2b. ORT **OpenVINO EP** | Apache-2.0 | mature for Intel | `onnxruntime-openvino` 1.24.1, Python 3.11–3.13, Linux + Windows; the Intel GPU needs Intel's OpenCL / Level Zero runtime, which the Flatpak runtime lacks | the same networks | Intel GPUs only, none on AMD. Its CPU path is a good CPU speed-up anywhere | small, Intel only |
| 2c. ORT **DirectML EP** | MIT | "sustained engineering" (new work moved to WinML) | `onnxruntime-directml`, Windows only | the same networks | any DX12 GPU on Windows (AMD, Intel, NVIDIA) | small, Windows only |
| 2d. ORT **MIGraphX EP** (the ROCm EP is deprecated) | MIT | maintained | `onnxruntime-migraphx` 1.27.1, Linux only, glibc 2.34+; needs a ROCm host | the same networks | only ROCm-supported AMD cards, the same ones PyTorch ROCm already covers. No help on APUs | small, no new coverage |
| 3a. SiftGPU (GLSL) | **educational / research / non-profit only** | archived (Oct 2024 fork); COLMAP's PyPI wheels have no GPU | needs an OpenGL context in a worker; GLSL 2008-era | SIFT + matching | fast in its day, but the licence rules it out | — |
| 3b. OpenCV OpenCL (T-API) | Apache-2.0 | `BFMatcher` has OpenCL kernels (`ocl_knnMatchSingle`); **SIFT has none** (`sift.dispatch.cpp` has no OpenCL path) | needs an OpenCL ICD (Mesa rusticl or the vendor's) inside the sandbox | matching only | matching on GPU if an ICD exists; extraction stays on the CPU | small, but patchy coverage |
| 3c. Kompute | Apache-2.0 | v0.7.0 badge, quiet project; Python 3.7–3.9 badge | a Python Vulkan layer next to ours | whatever we write | same kernels as option 1, plus per-dispatch overhead | worse than 1 (adds a dependency, does nothing new) |
| 3d. wgpu / WGSL | MIT / Apache-2.0 | mature | Rust toolchain in the build | whatever we write | no float atomics, limited subgroups (why vktrain chose GLSL) | worse than 1 |
| 4a. ROCm on APUs via `HSA_OVERRIDE_GFX_VERSION=10.3.0` | MIT-style | unsupported by AMD; reports mixed (works on ROCm 5.4 Deck guides; newer builds hang or show 0 devices) | already possible (`rocm6.4` profile, ~3 GB download) | everything PyTorch runs | if it works: GPU PyTorch, but within a 1–4 GB carve-out (Deck BIOS); OOM-prone | trivial to try; unsupportable |
| 4b. PyTorch Vulkan backend | BSD | **unmaintained**; moved to ExecuTorch (Android-focused, desktop experimental) | — | — | — | dead end |
| 4c. PyTorch XPU | BSD | supported for Intel Arc and Core Ultra graphics | `xpu` profile exists (Linux); wheels for Windows too | everything | good on Arc; older Iris Xe not covered | done (untested) |
| 4d. `torch-directml` | MIT | 0.2.5.dev, "Alpha", **no release since Sep 2024** | Windows (and WSL) | partial operator coverage, CPU fallbacks | uncertain | low effort, high risk |

Two findings change the picture:

- **pycolmap 4.x already ships ALIKED + LightGlue through ONNX.** COLMAP 4.0 added ALIKED extraction and LightGlue
  matching (for SIFT and ALIKED) through ONNX Runtime. 4.2 added CoreML as an EP on macOS. Our `pycolmap>=3.10`
  pin already allows 4.2.1, which has wheels for Python 3.10–3.14 (the GPU wheels are CUDA-only, as
  `pycolmap-cuda12`). On a CPU-only worker this may already beat PyTorch-CPU LightGlue. It is worth a 1-day check
  (stage 0) whatever we choose.
- **The processing runtime's Python 3.10 blocks current ONNX Runtime.** `onnxruntime` 1.32 and the WebGPU and
  OpenVINO packages need Python 3.11 or later. The app's own Python 3.14 could host them without the runtime
  download.

## Why option 1 over the alternatives

- **Coverage.** Vulkan is the only API that reaches every target with one code path: RADV on the Deck and
  610M (and GCN 3+ AMD), ANV for Intel, AMD's and Intel's Windows drivers, NVIDIA, and lavapipe for CI. ONNX
  Runtime needs one EP per vendor (OpenVINO, DirectML, MIGraphX). Only its WebGPU EP is cross-vendor, and it is beta.
- **It is the expensive part.** Brute-force matching is one GEMM-shaped kernel (4,096 × 4,096 × 128 ≈ 4.3 GFLOP
  per pair) plus row and column arg-min with the second-best distance. The Deck's GPU (8 RDNA 2 CUs, about 1.6
  TFLOPS FP32, twice that in packed FP16) does it in a few milliseconds, even at modest efficiency. The 610M (2 CUs,
  about 0.55 TFLOPS) does it in about 10–20 ms. No subgroup width is assumed, and no float atomics are needed.
- **Infrastructure exists.** `vkt::Context` (buffers, staging, embedded SPIR-V pipelines, the device list and
  probe), `blur.comp` (separable Gaussian, for the DoG pyramid), radix sort and scan (`radix_*`, `scan_*`, for
  compacting keypoints and top-k per image), the lavapipe CI container, the Flatpak module and the Windows plan.
  Adding a second executable keeps the same "subprocess, JSON lines" contract the worker already uses for vktrain.
- **No download, no fragile stack.** ROCm on APUs is unsupported and memory-limited, and the PyTorch Vulkan backend
  is dead. torch-directml has stalled, and DirectML is in maintenance. Our own kernels depend only on the Vulkan
  driver.
- **Licences are clean.** SIFT is ours to implement. SiftGPU is non-commercial, and SuperPoint's weights are
  non-commercial (Magic Leap licence), which already matters for a LightGlue path and would matter for any ONNX
  export of it.

**Risks.** SIFT parity takes care: sub-pixel and scale interpolation, orientation histograms and descriptor
trilinear binning need to land close to OpenCV or COLMAP SIFT, or solves get worse. SIFT + brute-force matching is
weaker than SuperPoint + LightGlue on low-texture or wide-baseline pairs. The 4080 keeps LightGlue, and
COLMAP/GLOMAP mapping stays on the CPU everywhere. Steam Deck VRAM is a 1 GB carve-out by default. Matching needs
only about 4 MB of descriptors per pair, so it fits easily.

## Staged plan

- **Stage 0, 1–2 days (independent of the choice):** measure pycolmap 4.2's ONNX ALIKED + LightGlue and its CPU
  SIFT against our OpenCV SIFT path on the Deck. Replace the per-match Python loop in `opencv_sift.match` with
  NumPy (a cheap CPU win). This sets the real CPU baseline.
- **Stage 1, about 1 week: Vulkan matcher.** `splatwork-vk-features match`: reads descriptors (RootSIFT float or
  FP16, any 128-d or 256-d descriptor), tiles the distance matrix in shared memory, keeps per-row and per-column
  best and second-best, then applies the mutual test and ratio test. Output is COLMAP's match format. It gets a CPU
  reference in the self-test (exact, on lavapipe) and the benchmark below. splatprep's `opencv_sift` backend calls it
  whenever PyTorch has no GPU and a Vulkan GPU is found (the CPU path stays as the fallback). **Go/no-go here.**
- **Stage 2, 3–4 weeks: Vulkan SIFT.** Gaussian pyramid (reusing `blur.comp`), DoG, 3×3×3 extrema with
  contrast and edge tests and quadratic refinement, keypoint compaction (scan), orientation histograms (36 bins,
  shared memory), 4×4×8 descriptors, RootSIFT, top-k by response (radix sort). Checked against OpenCV on
  synthetic images (repeatability, descriptor distance) and on the reference datasets (registered images,
  reprojection error, `--geometry` eval). The Linux Flatpak build comes first, then the Windows MSI build alongside
  vktrain's.
- **Stage 3 (separate decision): learned networks.** Either an ONNX Runtime WebGPU EP path (move the processing
  runtime to Python 3.11+, export ALIKED / DISK + LightGlue via LightGlue-ONNX, and Depth Anything V2 Small). Or
  LightGlue's transformer in our own GLSL (attention plus MLPs, no training needed, weights Apache-2.0). Choose from
  the stage 1–2 numbers and how much SIFT + brute force loses against LightGlue on our datasets.

## Prototype benchmark plan

**Data:** two of our reference datasets (one indoor, one outdoor drone) at 1,600 px, 4,096 keypoints, sequential
pairs (overlap 10) plus exhaustive pairs on a 60-image subset. Each machine records per-image extraction time,
per-pair matching time, end-to-end feature + match wall time, peak GPU memory, and the solve quality: registered
images, mean reprojection error, track length, and the `--geometry` eval.

| Machine | Runs |
|---|---|
| RTX 4080 (BensPC) | **baseline:** SuperPoint + LightGlue on CUDA (current default); COLMAP SIFT (CPU); Vulkan matcher on NVIDIA (sanity and the ceiling) |
| Radeon 610M (BensPC iGPU, `--device` index from `--list-devices`) | COLMAP SIFT (CPU, same Ryzen); Vulkan matcher on the 610M; stage 2: Vulkan SIFT |
| Steam Deck (Van Gogh, SteamOS, Flatpak) | COLMAP SIFT (CPU); pycolmap 4.2 ONNX ALIKED + LightGlue (CPU); Vulkan matcher; stage 2: Vulkan SIFT. Optional: the `rocm6.4` profile with `HSA_OVERRIDE_GFX_VERSION=10.3.0` and 4 GB UMA, for the record |
| Lavapipe (CI container) | correctness only: the matcher must equal the CPU reference bit-for-bit on match sets; SIFT within tolerances of OpenCV. Timing is logged, not gated |

**Pass criteria for stage 1:** on the Deck, matching at least 20× faster per pair than OpenCV `BFMatcher` on its
CPU, with identical match sets for the same descriptors (both are exact nearest neighbour; only the ratio-test
tie-breaking may differ). Feature + match wall time for the 300-image set comes in under 25 % of today's. For
stage 2: registered images and reprojection error within 2 % of COLMAP SIFT on both datasets.

**Harness:** extend `tools/splatprep/benchmark.py` with `--backend vk_sift` and `--matcher vk`, reusing its timing
and quality output, so all three machines produce one comparable table.

## Sources

- ONNX Runtime execution providers: [overview](https://onnxruntime.ai/docs/execution-providers/),
  [WebGPU EP](https://onnxruntime.ai/docs/execution-providers/WebGPU-ExecutionProvider.html),
  [DirectML EP (sustained engineering)](https://onnxruntime.ai/docs/execution-providers/DirectML-ExecutionProvider.html)
- PyPI: [onnxruntime](https://pypi.org/project/onnxruntime/),
  [onnxruntime-ep-webgpu](https://pypi.org/project/onnxruntime-ep-webgpu/),
  [onnxruntime-openvino](https://pypi.org/project/onnxruntime-openvino/),
  [onnxruntime-migraphx](https://pypi.org/project/onnxruntime-migraphx/),
  [torch-directml](https://pypi.org/project/torch-directml/), [pycolmap](https://pypi.org/project/pycolmap/)
- Models and licences: [LightGlue](https://github.com/cvg/LightGlue) (Apache-2.0; notes on SuperPoint's licence and
  speeds), [LightGlue-ONNX](https://github.com/fabio-sim/LightGlue-ONNX) (Apache-2.0),
  [SuperPoint licence](https://github.com/magicleap/SuperPointPretrainedNetwork/blob/master/LICENSE),
  [ALIKED](https://github.com/Shiaoming/ALIKED) (BSD-3), [DISK](https://github.com/cvlab-epfl/disk) (Apache-2.0),
  [Depth Anything V2 Small](https://huggingface.co/depth-anything/Depth-Anything-V2-Small) (Apache-2.0)
- COLMAP: [changelog](https://colmap.github.io/changelog.html) (4.0 ONNX ALIKED / LightGlue, 4.2 CoreML EP),
  [features](https://colmap.github.io/features.html),
  [SiftGPU licence](https://github.com/colmap/colmap/blob/main/src/thirdparty/SiftGPU/LICENSE),
  [SiftGPU fork (archived)](https://github.com/pitzer/SiftGPU)
- OpenCV source: [sift.dispatch.cpp](https://github.com/opencv/opencv/blob/4.x/modules/features2d/src/sift.dispatch.cpp)
  (no OpenCL), [matchers.cpp](https://github.com/opencv/opencv/blob/4.x/modules/features2d/src/matchers.cpp)
  (OpenCL `BFMatcher`), [T-API notes](https://github.com/opencv/opencv/wiki/OpenCL-optimizations)
- Vulkan frameworks: [Kompute](https://github.com/KomputeProject/kompute)
- PyTorch non-CUDA: [Vulkan workflow (unmaintained, see ExecuTorch)](https://docs.pytorch.org/tutorials/unstable/vulkan_workflow.html),
  [ExecuTorch Vulkan backend](https://docs.pytorch.org/executorch/stable/backends/vulkan/vulkan-overview.html)
- ROCm on APUs: [ROCm on a Steam Deck (ROCm 5.4.2, override 10.3.0)](https://element84.com/machine-learning/running-rocm-5-4-2-onnx-and-pytorch-on-a-steamdeck/),
  [ROCm issue #2527 (override not working)](https://github.com/ROCm/ROCm/issues/2527),
  [Debian report on gfx1033](https://lists.debian.org/debian-ai/2024/11/msg00061.html),
  [AMD: APU memory (GTT vs carve-out)](https://rocm.docs.amd.com/en/develop/how-to/system-optimization/rdna3-5.html)
