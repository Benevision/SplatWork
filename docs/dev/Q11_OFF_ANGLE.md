# Q11: off-angle views — measurement and results

Issue Q11 (ROADMAP section 5): splats look right from the capture direction and smear into needles, streaks and
see-through patches from other angles. This file holds the method, the baseline and every before / after table.

## Method: the holdout-sector score

`tools/eval/novel_views.py sector` (Mac headless CLI underneath):

1. A contiguous 40° sector of cameras is held out of training, together with the dataset's ordinary every-Nth
   held-out photos. Orbits (cameras look inwards) use the angle of the camera position round the scene; inside-out
   captures such as rooms use the angle of the view direction. Of all 40° sectors (5° steps) the one holding
   nearest 10 % of the cameras is used, so the choice is fixed per dataset.
2. The model is trained (`--train`, Balanced preset, adaptive strategy — MCMC is excluded until Q1 is fixed).
3. Scores, renders at 1600 px:
   - **sector PSNR / SSIM**: the held-out sector, i.e. real photos from an angle no training photo was taken from;
   - **normal PSNR / SSIM**: the ordinary held-out photos outside the sector (must not fall);
   - geometry indicators that need no ground truth, read from the splat file: `ratio > 10` (longest axis more than
     10 × the shortest), `needles` (longest more than 5 × the middle axis), mean opacity, share of splats under
     0.05 opacity, thickness along the splat's normal (median shortest axis in mm, and relative to the other two:
     1 = ball, 0 = flat disc).

```
python3 tools/eval/novel_views.py sector --dataset <dataset> --app mac/.build/release/SplatStudio \
    --out q11/van --name baseline [--config overrides.json]
python3 tools/eval/novel_views.py table q11/van
```

Scenes (not in the repository):

| scene | dataset | sector |
|---|---|---|
| van | `Capture 2026-11-16 12.19.31`, 431 LiDAR photos, 2400 px (the reported case) | 43 cameras, by position, centred 130° |
| bedroom | `Capture 2026-11-07 12.11.20`, 649 LiDAR photos, 1920 px | 65 cameras, by view direction, centred 145° |

Sector cameras on the van are 0.07–1.1 m from the nearest training camera. Some look at the van's interior, which
few other photos cover, so the sector score has a floor that no training change can lift.

Run-to-run noise (two seeds, van): about ±0.3 dB on both PSNR columns. Differences smaller than 0.5 dB are not
findings.

## Baseline (2026-11-17, Mac M4 Pro, Metal trainer, commit b3d2770)

| scene | sector PSNR | sector SSIM | normal PSNR | normal SSIM | splats | ratio > 10 | needles | mean opacity | opacity < 0.05 | thickness mm | thickness rel |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| van | 15.04 | 0.5669 | 19.99 | 0.6697 | 497,824 | 67.6% | 30.5% | 0.243 | 51.6% | 1.47 | 0.112 |
| bedroom | 18.37 | 0.7066 | 20.94 | 0.7429 | 800,000 | 31.7% | 16.3% | 0.425 | 26.6% | 1.74 | 0.267 |

## Finding: the Mac's adaptive strategy was deleting surfaces while it densified

The baseline's training log showed the real fault before any of the roadmap's fixes were tried: training PSNR
fell for as long as densification ran (bedroom 21.4 dB at step 3,000 → 15.9 dB at step 12,000) and jumped 4 dB in
the 500 steps after it stopped. Turning opacity resets off, or the bilateral grid off, did not change that
(`x_noreset`, `x_nogrid` below). Three Mac-only differences from gsplat's default strategy and the Vulkan trainer
were found; each was changed and measured on its own:

1. **Screen-size prune.** After the first opacity reset, every densify step deleted any splat that had been drawn
   wider than 10 % of a photo. On a close-range capture that is every splat on a surface a camera stood near.
   gsplat's default strategy has this rule switched off and the Vulkan trainer never had it. Removed; the
   world-size rule (larger than 10 % of the scene) stays.
2. **Evaluation and the colour safeguard ran on the step of an opacity reset** (steps 6,000 and 12,000 are
   multiples of both intervals). The just-reset splats render at 4–8 % brightness, the log showed held-out PSNR of
   5–10 dB, and the safeguard read it as "colour correction has taken over": it reset the bilateral grid at 6,000
   and switched it off at 12,000 on every adaptive run. Both now wait 500 steps after a reset.
3. **Opacity reset on the last densify step** (gsplat and Vulkan stop before it). With the presets' stop at 80 %
   of the run, a fifth of the run was left to rebuild every opacity from 0.01. No reset on that step now.

Van:

| run | sector PSNR | sector SSIM | normal PSNR | normal SSIM | splats | ratio > 10 | needles | mean opacity | opacity < 0.05 | thickness mm | thickness rel |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 15.04 | 0.5669 | 19.99 | 0.6697 | 497,824 | 67.6% | 30.5% | 0.243 | 51.6% | 1.47 | 0.112 |
| 1: no screen-size prune | 17.98 | 0.6303 | 23.00 | 0.7138 | 697,996 | 61.6% | 47.6% | 0.252 | 51.8% | 2.02 | 0.183 |
| 2: + no evaluation / colour safeguard for 500 steps after a reset | 17.74 | 0.6102 | 22.19 | 0.6636 | 735,901 | 61.2% | 48.3% | 0.234 | 55.9% | 2.01 | 0.185 |
| 3: + no reset on the last densify step (**new default behaviour**) | 17.84 | 0.6088 | 22.24 | 0.6601 | 758,114 | 63.0% | 49.0% | 0.442 | 28.0% | 1.85 | 0.178 |
| 3, seed 7 (noise check) | 18.09 | 0.6159 | 22.54 | 0.6627 | 763,809 | 62.3% | 47.5% | 0.448 | 27.5% | 1.85 | 0.179 |

Bedroom:

| run | sector PSNR | sector SSIM | normal PSNR | normal SSIM | splats | ratio > 10 | needles | mean opacity | opacity < 0.05 | thickness mm | thickness rel |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 18.37 | 0.7066 | 20.94 | 0.7429 | 800,000 | 31.7% | 16.3% | 0.425 | 26.6% | 1.74 | 0.267 |
| 1: no screen-size prune | 19.94 | 0.7247 | 22.36 | 0.7593 | 800,000 | 21.3% | 11.5% | 0.424 | 23.3% | 2.63 | 0.326 |
| 2: + no evaluation / colour safeguard for 500 steps after a reset | 19.72 | 0.7248 | 21.86 | 0.7601 | 800,000 | 20.5% | 11.4% | 0.424 | 22.5% | 2.67 | 0.332 |
| 3: + no reset on the last densify step (**new default behaviour**) | 19.68 | 0.7232 | 21.87 | 0.7605 | 800,000 | 21.6% | 11.8% | 0.539 | 17.6% | 2.60 | 0.324 |

Net (baseline → 3): van sector **+2.8 dB**, normal +2.3 dB; bedroom sector **+1.3 dB**, normal +0.9 dB. Change 1
carries the gain. Change 2 on its own scores lower than change 1 (van normal 23.00 → 22.19): the false alarm had
been switching the bilateral grid off, and these runs score higher without it — see "Bilateral grid" below. It is
kept because the alarm was wrong and the logged held-out score was meaningless on those steps. Change 3 does not
move PSNR; it halves the share of nearly transparent splats (van 56 % → 28 %, bedroom 23 % → 18 %).

Diagnosis runs on the old trainer (van):

| run | sector PSNR | sector SSIM | normal PSNR | normal SSIM | splats | ratio > 10 | needles | mean opacity | opacity < 0.05 | thickness mm | thickness rel |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 15.04 | 0.5669 | 19.99 | 0.6697 | 497,824 | 67.6% | 30.5% | 0.243 | 51.6% | 1.47 | 0.112 |
| opacityResetInterval 0 | 15.17 | 0.5458 | 18.57 | 0.5844 | 799,971 | 60.2% | 28.4% | 0.671 | 18.2% | 1.64 | 0.153 |
| bilateralGrid off | 15.12 | 0.5661 | 19.79 | 0.6653 | 511,267 | 67.2% | 27.9% | 0.223 | 56.8% | 1.49 | 0.108 |

## The roadmap's five fixes, one at a time (van, on top of change 3)

| run | sector PSNR | sector SSIM | normal PSNR | normal SSIM | splats | ratio > 10 | needles | mean opacity | opacity < 0.05 | thickness mm | thickness rel |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| repaired trainer | 17.84 | 0.6088 | 22.24 | 0.6601 | 758,114 | 63.0% | 49.0% | 0.442 | 28.0% | 1.85 | 0.178 |
| (a) needle limit 4, anisotropyReg 1 | 17.82 | 0.6074 | 22.50 | 0.6648 | 647,440 | 40.9% | 0.0% | 0.493 | 24.2% | 2.21 | 0.207 |
| (a) flatnessReg 0.005 | 18.04 | 0.6105 | 22.34 | 0.6530 | 765,206 | 94.0% | 51.6% | 0.394 | 31.1% | 0.03 | 0.002 |
| (b) depthWeight 0.5 (LiDAR) | 17.96 | 0.6106 | 22.22 | 0.6539 | 799,999 | 57.6% | 44.2% | 0.382 | 32.6% | 2.11 | 0.193 |
| (c) SH degree 1 | 17.59 | 0.6035 | 21.84 | 0.6401 | 734,484 | 64.1% | 47.3% | 0.432 | 29.5% | 1.81 | 0.169 |
| (c) penalty on higher SH bands, 10 | 17.72 | 0.6107 | 22.36 | 0.6703 | 771,773 | 63.6% | 47.5% | 0.426 | 29.0% | 1.77 | 0.173 |
| (d) solid-or-gone opacity 0.05 | 18.05 | 0.6150 | 22.36 | 0.6583 | 467,429 | 63.7% | 47.8% | 0.845 | 9.9% | 1.75 | 0.173 |
| (e) contribution prune of the repaired run (Auto clean, 0.25) | 17.83 | 0.6087 | 22.19 | 0.6591 | 455,410 | 62.9% | 46.7% | 0.639 | 4.8% | 1.84 | 0.174 |

None moves the sector score by more than the ±0.3 dB noise, so **none becomes a default** (the bar was +1 dB on
the van). What each did:

- (a) A needle limit of 4 removes every needle at no cost in PSNR; flatness makes splats paper-thin (0.03 mm)
  without helping. On the old trainer flatness 0.1 and 1 lost 0.3–0.7 dB.
- (b) Only the per-view LiDAR depth weight was tested. A PGSR-style term that reprojects rendered depth and
  normals into neighbouring views was **not implemented**.
- (c) Lower SH degree and a penalty on the higher bands change nothing: view-dependent colour is not what breaks
  these views. The penalty was removed from the code again.
- (d) Solid-or-gone opacity at 0.05: same scores with 38 % fewer splats, mean opacity 0.45 → 0.85, nearly
  transparent splats 28 % → 10 %. Kept as an option, off by default, Metal trainer only
  (`opacityBinaryReg`). At 1 and 10 it deletes the scene (12,368 and 864 splats left).
- (e) Pruning by contribution across the training photos removes 40 % of the splats with no change in any score:
  a free size cut (issue Q2), not an off-angle fix.

The same runs on the old trainer, for the record (van):

| run | sector PSNR | sector SSIM | normal PSNR | normal SSIM | splats | ratio > 10 | needles | mean opacity | opacity < 0.05 | thickness mm | thickness rel |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 15.04 | 0.5669 | 19.99 | 0.6697 | 497,824 | 67.6% | 30.5% | 0.243 | 51.6% | 1.47 | 0.112 |
| (a) needle limit 4, anisotropyReg 1 | 15.36 | 0.5687 | 20.24 | 0.6740 | 498,575 | 54.2% | 0.0% | 0.273 | 49.7% | 1.78 | 0.137 |
| (a) flatnessReg 0.1 | 14.79 | 0.5711 | 19.18 | 0.6631 | 467,763 | 100.0% | 30.3% | 0.243 | 48.2% | 0.00 | 0.000 |
| (a) flatnessReg 1 | 14.31 | 0.5477 | 19.11 | 0.6536 | 456,005 | 100.0% | 29.3% | 0.227 | 51.0% | 0.00 | 0.000 |
| (b) depthWeight 0.5 | 15.41 | 0.5734 | 19.26 | 0.6578 | 800,000 | 66.4% | 24.3% | 0.168 | 58.4% | 1.23 | 0.095 |
| (c) SH degree 1 | 14.78 | 0.5613 | 19.83 | 0.6654 | 606,916 | 72.0% | 28.8% | 0.217 | 55.6% | 1.10 | 0.083 |
| (c) SH band penalty 10 | 14.42 | 0.5569 | 19.34 | 0.6632 | 588,694 | 73.6% | 27.4% | 0.205 | 56.7% | 0.97 | 0.070 |
| (d) solid-or-gone 1 | 14.36 | 0.5662 | 18.74 | 0.6464 | 12,368 | 85.5% | 4.6% | 0.904 | 7.2% | 0.29 | 0.004 |
| (d) solid-or-gone 10 | 13.04 | 0.5244 | 16.88 | 0.6027 | 864 | 83.7% | 2.5% | 0.976 | 0.8% | 0.19 | 0.001 |

## Bilateral grid (for issue Q7; not changed here)

On the repaired trainer, adaptive, `bilateralGrid` off:

| run | sector PSNR | sector SSIM | normal PSNR | normal SSIM | splats | ratio > 10 | needles | mean opacity | opacity < 0.05 | thickness mm | thickness rel |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| van, grid on | 17.84 | 0.6088 | 22.24 | 0.6601 | 758,114 | 63.0% | 49.0% | 0.442 | 28.0% | 1.85 | 0.178 |
| van, grid off | 17.79 | 0.6292 | 23.30 | 0.7201 | 674,432 | 61.7% | 46.2% | 0.459 | 26.9% | 1.90 | 0.179 |
| van, grid on, seed 7 | 18.09 | 0.6159 | 22.54 | 0.6627 | 763,809 | 62.3% | 47.5% | 0.448 | 27.5% | 1.85 | 0.179 |
| van, grid off, seed 7 | 18.22 | 0.6363 | 23.51 | 0.7279 | 712,459 | 61.0% | 45.5% | 0.456 | 26.9% | 1.93 | 0.181 |

| run | sector PSNR | sector SSIM | normal PSNR | normal SSIM | splats | ratio > 10 | needles | mean opacity | opacity < 0.05 | thickness mm | thickness rel |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| bedroom, grid on | 19.68 | 0.7232 | 21.87 | 0.7605 | 800,000 | 21.6% | 11.8% | 0.539 | 17.6% | 2.60 | 0.324 |
| bedroom, grid off | 19.82 | 0.7222 | 22.33 | 0.7597 | 800,000 | 22.5% | 11.7% | 0.545 | 18.0% | 2.64 | 0.319 |

Ordinary held-out photos score about 1 dB higher on the van (SSIM 0.66 → 0.72) and 0.5 dB on the bedroom with
the grid off; the sector is unchanged. The default was left alone: it is Q7's decision, and held-out photos are
scored without any correction, which favours "off".

## Where Q11 stands

Better, not fixed. Sector views on the van are still 4.4 dB below ordinary held-out views and still show streaks
and smeared lettering. Not yet tried: multi-view depth / normal consistency between neighbouring views (b), a run
with the 43 nearest cameras restored to see how much of the sector gap is missing coverage, the High preset, and
the gsplat and Vulkan trainers on the same sector split.
