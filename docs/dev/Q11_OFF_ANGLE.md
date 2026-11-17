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
