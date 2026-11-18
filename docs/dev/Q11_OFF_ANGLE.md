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

## Views from below the capture (2026-11-18)

A van splat trained with the repaired trainer (build 306) still smears when viewed from near the ground. The sector
score did not show this: it removes a slice of angles but keeps the same height band. `novel_views.py sector --by low`
(or `high`) holds out the lowest (highest) `--fraction` of the cameras instead. Van, re-prepared dataset (dense
points, refined poses), lowest 65 of 431 cameras held out (`--fraction 0.15`), Balanced, adaptive, Metal:

| run | low views PSNR | SSIM | normal PSNR | SSIM | splats | needles | mean opacity |
|---|---:|---:|---:|---:|---:|---:|---:|
| current trainer | 16.76 | 0.5748 | 22.07 | 0.6601 | 800,000 | 39.5% | 0.469 |
| (a) needle limit 4 | 16.84 | 0.5873 | 22.07 | 0.6577 | 734,571 | 0.0% | 0.494 |
| (c) SH degree 1 | 16.58 | 0.5822 | 21.78 | 0.6549 | 781,693 | 40.4% | 0.463 |
| (b) depth weight 0.5 | 16.91 | 0.5877 | 21.97 | 0.6554 | 800,000 | 36.0% | 0.389 |
| (d) solid-or-gone 0.05 | 16.85 | 0.5853 | 22.21 | 0.6633 | 498,866 | 41.4% | 0.834 |
| depth spread 0.05 | 16.71 | 0.5861 | 22.10 | 0.6610 | 800,000 | 40.3% | 0.478 |
| depth spread 0.5 | 14.76 | 0.5384 | 21.42 | 0.6454 | 501,805 | 37.8% | 0.506 |
| depth spread 2 | 9.39 | 0.2782 | 16.94 | 0.5862 | 413,238 | 21.4% | 0.452 |

Low views are 5.3 dB below ordinary held-out photos, and no option closes the gap (noise is about 0.3 dB). Removing
every needle changes the score by 0.08 dB, so needles are not what this score punishes.

Depth spread was a new term: sum of weight x |splat depth - sensor depth| along each pixel's ray, so that surfaces
are one layer at the LiDAR depth and not a see-through stack that only blends to it. No effect at 0.05, harmful
from 0.5. The code is removed (it reached `main` for one commit, 5b13da1, off by default, when another session's
commit picked up the working tree).

### How much of the low-view gap is missing coverage

`--thin 2` keeps only every other camera of the band held out and returns the rest to training. The same 33 low
views, van, current trainer:

| training set | 33 low views PSNR | SSIM | normal PSNR |
|---|---:|---:|---:|
| all 65 low cameras held out | 16.74 | 0.5733 | 21.78 |
| the other 32 low cameras train | 20.22 | 0.6512 | 22.33 |

Photos from the same height recover 3.5 dB of the 5.6 dB gap to ordinary held-out photos; about 2 dB remains.
Most of the smear from below is therefore surfaces and angles no training photo shows, which a trainer change
cannot invent: the remedy is capture (a low pass and a high pass) and telling the user when a height band is
missing. The roadmap fixes were tested against the part that coverage explains, which is why none registered.

### Neighbouring-view photo consistency (b), on the thinned split

Tried in the Metal trainer and removed again. The rendered depth of each pixel places it in 3D; that point is
projected into one of the four nearest training photos looking within 30 degrees, and the difference between the two
photos there (grey, local 5 x 5 mean removed so exposure drops out; differences over 0.1 skipped as occlusion) is a
loss whose gradient goes to the rendered depth. This is the multi-view photometric idea of PGSR, per pixel; PGSR's
plane patches, NCC and forward-backward geometric check were not built. Van, the same 33 low views:

| run | 33 low views PSNR | SSIM | normal PSNR | SSIM |
|---|---:|---:|---:|---:|
| current trainer | 20.22 | 0.6512 | 22.33 | 0.6684 |
| weight 0.3 | 20.35 | 0.6465 | 22.11 | 0.6673 |
| weight 1 | 20.32 | 0.6528 | 22.02 | 0.6630 |
| weight 10 | 20.47 | 0.6598 | 22.15 | 0.6672 |

All within the 0.3 dB noise, including a weight 100 times the depth loss's. On captures with LiDAR depth the
surface position is already fixed, so there is little left for this term to correct. Not tested on a photo-only
capture, where it has more to do.

### Photogrammetry on the same photos (Apple Object Capture, full detail)

To see what a mesh method makes of the same views: the 348 training photos of the thinned split, photos only
(no LiDAR depth, no ARKit poses), 20 minutes on the M4 Pro.

- Its structure-from-motion registered 295 of 348 photos, and only 169 of those agree with the ARKit poses (median
  4 cm after a similarity fit). The other 126, mostly the first 130 photos, are placed up to 177 degrees wrong:
  the two sides of the van were confused. The mesh there is ghost geometry.
- Where its poses are right (21 of the 33 held-out low views): matte surfaces (the bags, floor and fittings inside
  the van) are sharper in the mesh than in the splat; the glossy white and green bodywork is torn, doubled or
  missing in the mesh and whole in the splat.
- PSNR on those 21 views: mesh 11.8 dB, splat 19.5 dB. The mesh number is dominated by the 4 cm alignment error
  and its unlit texture, so it only says the mesh is not a pixel-accurate renderer; the comparison is by eye.

What this says about Q11: a mesh has one opaque surface, so matte detail holds from any angle, and that is where
the splat is soft from below (blurred, not streaked, on the interior views). Reflective bodywork is the reverse.
The splat's softness on those views points at detail (splat budget and resolution on rarely seen surfaces) more
than at geometry.

### The same three tests on a room with little glass (2026-11-18)

`Capture 2026-11-17 17.22.00`: a bedroom, 445 LiDAR photos, lowest 67 cameras as the band, Balanced, adaptive, Metal.

| run | 34 low views PSNR | SSIM | normal PSNR | SSIM | needles |
|---|---:|---:|---:|---:|---:|
| all 67 low cameras held out | 17.84 | 0.5636 | 20.67 | 0.6566 | 6.5% |
| the other 33 low cameras train | 20.05 | 0.6322 | 20.90 | 0.6593 | 6.2% |
| + neighbouring-view consistency, weight 1 | 20.04 | 0.6322 | 20.86 | 0.6588 | 6.0% |

- Coverage again: 2.2 dB of the 2.8 dB gap returns with photos from that height, leaving 0.85 dB.
- Neighbouring-view consistency: no change (0.01 dB).
- Object Capture (359 photos, 22 minutes): 352 registered, 297 agree with the ARKit poses to a median of 1.4 cm,
  so its camera solve mostly works here, unlike on the van. On the 34 low views it scores 16.5 dB / SSIM 0.519
  against the splat's 20.2 dB / 0.675 (same caveat: alignment and unlit texture). By eye the mesh keeps fine
  texture the splat blurs (rug pattern, concrete floor, cables) and bends straight edges and flat screens.

Both scenes agree: what is left after coverage is softness, not streaks or needles (6 % needles here), and a
textured mesh shows that the photos contain that detail from these angles.

### Splat budget on the low views (2026-11-18)

Same thinned splits (half the low band trains, the other half is scored), Metal, adaptive.

| scene | run | low views PSNR | SSIM | normal PSNR | SSIM | splats | minutes |
|---|---|---:|---:|---:|---:|---:|---:|
| van | Balanced (cap 800,000) | 20.22 | 0.6512 | 22.33 | 0.6684 | 800,000 | 13.7 |
| van | Balanced, cap 2,000,000 | 20.33 | 0.6604 | 22.02 | 0.6672 | 815,515 | 9.1 |
| van | High preset (30,000 steps, 1600 px, cap 1,500,000) | 20.23 | 0.6455 | 22.05 | 0.6586 | 708,727 | 21.4 |
| bedroom | Balanced (cap 800,000) | 20.05 | 0.6322 | 20.90 | 0.6593 | 800,000 | 10.6 |
| bedroom | Balanced, cap 2,000,000 | 19.99 | 0.6387 | 20.82 | 0.6623 | 2,000,000 | 14.9 |
| bedroom | High preset | 20.02 | 0.6386 | 20.79 | 0.6629 | 1,499,999 | 30.5 |

- No gain: every difference is inside the 0.3 dB run-to-run noise, on low views and on normal held-out views.
- On the van the cap was not the limit: with room for 2,000,000 the trainer stopped at 815,515, and High ended
  at 708,727. In the bedroom it filled 2,000,000 and scored the same.
- High costs two to three times the training time for no held-out gain on either scene.
- So the softness on unseen views is not a shortage of splats, steps or training resolution. With coverage the
  low views are 2.0 dB (van) and 0.9 dB (bedroom) below ordinary held-out views, which are themselves soft
  (22 and 21 dB) while training views reach 24 to 25 dB: the gap is fit-to-photos versus new views in general.

### The same low-view split on the other two trainers (2026-11-18)

Same training photos and held-out views, Balanced settings sent explicitly (15,000 steps, 1400 px, cap 800,000,
growth stops at 12,000, adaptive). gsplat and Vulkan ran on the RTX 4080 (SplatWork for Linux 1.7.18); all
splats were scored by the Mac renderer. One run each.

| scene | trainer | low views PSNR | SSIM | normal PSNR | SSIM | splats | needles | minutes |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| van | Metal | 20.22 | 0.6512 | 22.33 | 0.6684 | 800,000 | 39.5% | 13.7 |
| van | gsplat | 19.85 | 0.6542 | 22.15 | 0.7019 | 480,577 | 35.1% | 6.3 |
| van | Vulkan | 21.21 | 0.6755 | 22.78 | 0.7156 | 808,111 | 22.8% | 18.3 |
| bedroom | Metal | 20.05 | 0.6322 | 20.90 | 0.6593 | 800,000 | 6.2% | 10.6 |
| bedroom | gsplat | 20.75 | 0.6361 | 21.59 | 0.6614 | 826,234 | 8.0% | 7.1 |
| bedroom | Vulkan | 20.82 | 0.6377 | 21.51 | 0.6612 | 849,535 | 7.1% | 23.0 |

- The low-view softness is in all three trainers: none comes within 1.5 dB of its own normal held-out score on
  the van, and SSIM on the low views stays at 0.64 to 0.68 everywhere.
- The Metal trainer is behind on three of four comparisons by more than noise: 1.0 dB behind Vulkan on the van
  low views (and 0.03 to 0.05 SSIM behind both on normal views), 0.7 dB behind both in the bedroom. On the van
  Vulkan also makes far fewer needles (23 % against 40 %).
- gsplat stopped at 480,577 splats on the van, so the three trainers do not grow alike under one specification
  (rule 5). The caps are also not exact on the PC trainers (808,111 to 849,535 against 800,000).

### Why Metal was behind: two Mac-only behaviours (2026-11-18)

Reading the Vulkan and Metal growth routines side by side: cloning, splitting and pruning follow the same rules.
Two things differ.

1. **Opacity reset.** Metal pulls every opacity down to 0.01 every 3,000 steps. Vulkan does not unless
   `--opacity-reset` is given, and gsplat's default strategy never does (its reset test is a chained comparison
   that is always false; noted in `linux/vktrain/src/trainer.cpp`).
2. **Colour correction with locked exposure.** All three trainers switch per-photo colour correction off when
   every photo has the same exposure (SplatWork Capture). On the Mac that rule covers `appearanceCorrection`
   only: `bilateralGrid` is on by default on the Mac (off by default in the other two) and keeps running.

Metal with each removed, same thinned low-view splits, one run each:

| scene | run | low views PSNR | SSIM | normal PSNR | SSIM | needles |
|---|---|---:|---:|---:|---:|---:|
| van | Metal as shipped | 20.22 | 0.6512 | 22.33 | 0.6684 | 39.5% |
| van | no resets (`opacityResetInterval` 0) | 20.61 | 0.6651 | 22.47 | 0.6811 | 22.6% |
| van | grid off | 20.77 | 0.6663 | 22.73 | 0.7093 | 37.9% |
| van | grid off, no resets | 21.24 | 0.6795 | 22.72 | 0.7148 | 22.1% |
| van | Vulkan | 21.21 | 0.6755 | 22.78 | 0.7156 | 22.8% |
| bedroom | Metal as shipped | 20.05 | 0.6322 | 20.90 | 0.6593 | 6.2% |
| bedroom | no resets | 20.03 | 0.6295 | 20.83 | 0.6568 | 6.3% |
| bedroom | grid off | 20.70 | 0.6339 | 21.51 | 0.6593 | 6.5% |
| bedroom | grid off, no resets | 20.68 | 0.6302 | 21.37 | 0.6573 | 6.5% |
| bedroom | Vulkan | 20.82 | 0.6377 | 21.51 | 0.6612 | 7.1% |

- With both removed Metal matches Vulkan on both scenes (within 0.15 dB).
- The grid is worth 0.55 to 0.65 dB on low views and 0.4 to 0.6 dB on normal held-out views on both scenes.
- Resets are worth 0.4 to 0.5 dB on the van's low views and nothing in the bedroom, and they are what makes the
  van's needles (39.5 % → 22 %). `opacityResetInterval` 1,000,000 scores the same as 0.
- Neither is changed here: each is a default change (one per commit, benchmark first), and the grid is Q7's
  decision. Proposed order: (1) the locked-exposure rule also switches the grid off on the Mac, which only brings
  Metal into line with the other two; (2) resets off in Metal's adaptive strategy, or the three trainers agree on
  one behaviour.

## Research review (2026-11-18)

Seventeen documents supplied by the user were read against the measurements above (16 papers and surveys, and
three proceedings volumes of 100 papers, of which five were relevant). Numbers are the papers' own and were not
reproduced. Most of the set is feed-forward prediction, compression, dynamic scenes or surface extraction and
does not bear on Q11. What does, in the order worth testing:

| # | Idea | Source | Paper's evidence | Cost | Fit to our measurements |
|---|---|---|---|---|---|
| 1 | Opacity decay (× 0.9995 every 50 steps while growing) in place of the reset | Hahlbohm et al., perspective-correct 3DGS; StopThePop | Their growth fails without it | One line per trainer | Resets make the van's needles; decay may keep the reset's clean-up without them |
| 2 | Prune by contribution: drop splats whose largest α × transmittance over all training pixels is under 0.01 | RadSplat (Niemeyer et al.) | 26.14 → 26.17 dB with a third fewer splats | One counting pass; Auto clean already has the census | 15 to 26 % of our splats are under opacity 0.05 |
| 3 | Several photos per step, gradients summed | MVGS (Du et al.) | 28.69 → 29.26 dB from this alone; up to 8× training time | Training-loop change | Aimed at the 2 to 3 dB between training and held-out views |
| 4 | 3D size filter (no splat smaller than the training photos can resolve) | Mip-Splatting, via both surveys | Fixes artefacts when the view distance changes | Low; already roadmap section 6 | Low views are closer to the floor than any training photo |
| 5 | Edge-aware photo term and edge-weighted normal loss | VA-GS (Li et al.) | +0.30 dB, LPIPS −0.040 on Mip-NeRF 360 for the whole method | Loss terms | Softness; their flattening penalty hurt novel views, so also measure ours off |
| 6 | Uncertainty channel fitted after training | Galappaththige et al. | Correlation 0.37 with held-out error | One extra channel | Does not improve the splat; could drive the coverage warning |
| 7 | Perspective-correct splat evaluation and sorted first-16 blending | Hahlbohm et al. | 24.62 → 24.81 dB; the gain is in motion and at close range | New rasterisers in three trainers and two viewers | Close-range, wide-angle captures are its case; largest job here |

Not worth testing for Q11: feed-forward predictors (SplatWeaver, Uni3R, MeshSplat, SparSplat, VolSplat, MVSplat,
G3Splat), compression (Niedermayr et al.), dynamic scenes (GS-Morph), the NeRF teacher in RadSplat. VA-GS's
multi-view patch loss is the term already measured here with no gain. Chougule (2025) reports the same symptom
(renders far from the capture poses are blurred) and no fix. Licences: most of these build on the Inria code or
do not state a licence, so only the papers' equations are used.

## Where Q11 stands

Better, not fixed. Sector views on the van are still 4.4 dB below ordinary held-out views and still show streaks
and smeared lettering. For views from a height the capture never visited, most of the gap is missing coverage
(3.5 dB van, 2.2 dB bedroom); neighbouring-view consistency, a larger splat budget and the High preset do not
move what is left. The other two trainers show the same softness. Metal was 0.7 to 1.0 dB behind them because of
opacity resets and the colour grid running on locked-exposure captures; both are proposed default changes. Not yet tried: a photo-only capture, and
anything aimed at held-out views in general (pose and exposure refinement at test time, the Q7 bilateral grid).
