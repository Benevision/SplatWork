# SplatWork plan and roadmap

Written 2026-11-17 from `main` at `b275a1e`, after a full review of the code, the history (298 commits in 11 days),
the open issues and current research. It replaces the release-only roadmap of 2026-11-15; that plan's phases are
kept as milestones 7 and 8. [STATUS.md](STATUS.md) says what is tested today.

This file is the working plan for every development session. Read **Rules** before changing anything, pick work
from the **first unfinished milestone**, and update the **Issue register** and the tick boxes in the same commit as
the work.

## 1. The goal

SplatWork is an easy but powerful Gaussian-splat suite: the best result the research allows, trained as fast as the
hardware allows, with processing shared across the machines on a network, plus a few unusual sources (drone, 360°,
stereo, home video). The comparison is Spirula Studio, LichtFeld Studio and Postshot.

"On track" means all four of these hold, and each has a number in `tools/eval/baselines.json` (milestone 0):

| Pillar | Measured by |
|---|---|
| Quality | Held-out PSNR / SSIM / LPIPS and a novel-view score on the benchmark scenes; photos placed by the solve |
| Speed | Minutes from photos to a finished splat per preset; steps per second; solve time; peak memory |
| Easy | A first-time user gets a good splat with no settings; no preset can produce a broken result |
| Networked | A block run on N devices is faster than one device and scores within 0.3 dB of a single-device run |

## 2. Review verdict

**Strong:** the Metal trainer is GPU-resident and fast; defaults carry their measurements in comments; failure
safeguards (colour guard, memory guard, pose sanity) are thoughtful; the worker isolates trainers in subprocesses
with atomic writes and resumable jobs; the Vulkan kernels are checked against a double-precision reference;
STATUS.md is honest about what is untested.

**What has gone wrong:**

1. **Quality stopped being measured.** Held-out numbers appear in commits until 2026-11-09; after that almost
   none. There is no regression benchmark for quality or speed, in CI or by hand. Haze, floaters, block seams and
   novel-view consistency have only ever been judged by eye. Four options shipped "off, to be measured"
   (progressive resolution, decoupled SSIM, error-driven densify, background shell) and never were.
2. **The same problems are fixed repeatedly.** Colour correction taking over (4 fixes, still open), camera solves
   (about 12), memory exhaustion (5 incidents, each patched in one stage), GPU queueing (7 Linux releases in 3
   days, one fixing a regression from another), capture blur (5).
3. **Breadth outran depth.** Five apps, three hand-ported trainers (Metal, gsplat, Vulkan) with three sets of
   defaults and no test that they agree. Windows, the Vulkan trainer, ROCm, RTX 50, stereo capture and
   multi-machine block runs shipped without a written real-hardware run. About 49 Linux versions in 11 days.
4. **Networked training is the weakest pillar where it should be the signature one.** Blocks are merged by a hard
   crop on splat centres, each block normalises colour on its own, outer blocks keep everything to infinity, and
   nothing has measured a block run against a single-device run.
5. **Security is further behind than the old roadmap said:** the 2026-11-09 scan has 55 findings (2 high, 29
   medium, 24 low); 5 are fixed, about 50 open. The old plan counted 25.

## 3. Rules (every session)

1. **Measure before and after.** Any change to training, solving, defaults or presets runs the benchmark
   (milestone 0) and puts the before / after table in the commit message. No number, no merge. Never judge
   quality by eye alone.
2. **One default change per commit**, never flipped and reverted the same day. Options that have no measurement
   are measured or deleted, not left "off, to be measured".
3. **Experimental means off by default** until there is a written run on real hardware in `docs/dev/`. Say
   "untested on real hardware" in the UI text and STATUS.md until then.
4. **Scope freeze until milestones 0–2 are done:** no new platforms, capture sources or trainers. Bug fixes and the
   milestone work only. New ideas go in section 8, not into code.
5. **Three trainers, one specification.** A training change lands in the shared defaults and all three trainers in
   the same piece of work, or it is explicitly marked single-trainer in the settings catalogue.
6. **Fix the class, not the instance.** For a repeat problem (memory, uploads, stuck jobs, colour, solves), find
   the earlier fixes in the Issue register first and replace them with one mechanism.
7. **Releases:** Linux versions are batched (at most one a day, normally weekly) after a soak run of a real mixed
   queue on the RTX 4080. `main` must build every target, including the iPhone app.
8. Standing project rules still apply: push every commit; never quit or replace the running Mac app (use
   `mac/scripts/update_local.sh`); the fixed identifiers in STATUS.md never change; Spirula and LichtFeld are
   GPL-3.0 and most paper code derives from the non-commercial Inria code — take ideas from papers, reference only
   Apache-2.0 code (gsplat, NVIDIA 3dgrut); never commit footage; ask before submitting jobs to the PC while the
   user's own jobs run.

## 4. Milestones

Do them in order. Each has a "done when" that can be checked.

### Milestone 0 — Measure (first; about a week)

- [ ] **Benchmark suite** `tools/eval/bench.py`: four fixed scenes from `Sample datasets/` (Bedroom LiDAR capture,
      the drone survey, one phone video, one 360° clip) plus the van capture that exposed the MCMC collapse. For
      each: solve, train with each preset, on each trainer available on the machine.
- [ ] It records held-out PSNR / SSIM / LPIPS, the novel-view score (`tools/eval/novel_views.py`: holdout-sector
      PSNR, mean opacity, fraction of splats under 0.05 opacity), photos placed, solve time, training time, steps
      per second, splat count, peak RAM and VRAM.
- [ ] `tools/eval/baselines.json` committed with today's numbers for Mac (Metal) and the RTX 4080 (gsplat, Vulkan).
      A `--check` mode fails when quality drops more than 0.3 dB or time rises more than 10 %.
- [ ] Runs by hand before every default change and release; nightly on the RTX 4080 when the queue is idle.
- [ ] **Divergence guard** in all three trainers: stop and report (or roll back to the last checkpoint) when
      training PSNR falls more than 4 dB below its running best, or when more than a set share of splats leave the
      scene bounds. A collapsed run must never be exported as a finished splat.
- [ ] Measure or delete the four unmeasured options.

Done when: the baseline table exists for both machines and `--check` passes twice in a row.

### Milestone 1 — Stop the known quality losses (1–2 weeks)

Each item is a known defect with evidence; fix, measure, tick.

- [ ] **MCMC collapse above about 1 M splats** (Issue Q1): see the register for evidence and the order of tests.
      Then choose the cap from the scene (points, photos, pixels) so a preset can't ask for 3 M splats on a scene
      that needs 400 k.
- [ ] **Off-angle views break up** (Q11, the user's top priority with Q1): surfaces that look right from the
      capture direction smear into needles and see-through streaks from other angles. First make it measurable
      (holdout-sector PSNR and a needle count in the benchmark), then test in this order, one change each:
      (a) splats flattened onto surfaces (flatness is weighted 0 today) with a tighter long-axis limit;
      (b) depth and normal consistency between neighbouring views, PGSR style, using the LiDAR depth where it
      exists; (c) view-dependent colour held back (lower SH degree until late, a penalty on the higher bands) so
      colour can't stand in for geometry; (d) opacity pushed to solid or gone late in training; (e) pruning by
      contribution across views. Windows and other reflective surfaces are expected to stay the hardest case.
- [ ] **Fog in exports** (Q2): run the invisible-splat census (Auto clean) automatically at the end of training
      and after a block merge; the only prune today is opacity above 0.005.
- [ ] **Presets leave too little refinement** (Q3): `densifyStop` stored as a fraction of the steps (about 0.5) in
      every preset; the Mac inspector rescales it when steps change, as the CLI already does.
- [ ] **Resume loses optimiser state on the Mac** (Q4): checkpoint Adam moments, step count and colour grids.
      Linux: write resume state every few minutes, not only on pause.
- [ ] **Remote solve differs from local** (Q5): one recipe on both sides (the Mac uses COLMAP SIFT locally but
      SIFT + LightGlue + pycolmap remotely); return the solve stats from the worker (`processing.py` reads a
      `stats` key that `run_request` never returns); show the alignment and refinement lines on the Mac.
- [ ] **Pose refinement can move single frames 20–28 cm** (Q6): add the capture pose as a prior in the bundle
      adjustment, or keep the capture pose for frames that move more than 3× the median; record per-frame moves.
- [ ] **Colour correction taking over** (Q7): reproduce on the van and drone sets with the benchmark; decide on
      one design (see milestone 3, appearance) instead of a fifth patch.
- [ ] Held-out photos are folded into the last steps of training (they are never trained on today).
- [ ] Mac release builds use `-O`, not `-Ounchecked`; fix security finding F1 (symlinks in a worker's output zip
      can write anywhere on the Mac).
- [ ] The worker's `Region` docstring matches the code (splats are not confined to the region).

Done when: every item is ticked with numbers, and the van capture trains to within 0.5 dB of its best known result
on every strategy and trainer.

### Milestone 2 — One trainer specification (1–2 weeks)

- [ ] One defaults file (JSON) read or generated into `TrainingConfig.swift`, `config.py` and `config.cpp`; a test
      on each side fails if a default differs.
- [ ] Golden-config tests: the same small synthetic scene and seed through all three trainers; losses and
      gradients agree within tolerance. The Vulkan self-test already has the reference; extend it to gsplat and
      Metal.
- [ ] Cross-trainer parity in the benchmark: the same scene scores within 0.3 dB on Metal, gsplat and Vulkan. A
      trainer that can't hold parity is labelled experimental and is never a default.
- [ ] A Mac test target (`swift test`) in CI: parsers, block split and merge, config decoding (a wrong type must be
      reported, not silently replaced by the default), upload batching, relocation maths.
- [ ] Split the 700-line `train()` in `trainer.py` and the scheduler out of `server.py`'s JobManager so loss terms
      and queue logic can be tested alone.

Done when: a training change can't land in one trainer without a failing test in the others.

### Milestone 3 — Quality from current research (3–4 weeks)

Adopt in this order; each is one measured change. Reimplement from the papers.

- [ ] **One capped hybrid densification strategy** replacing the MCMC / adaptive choice: error-guided growth,
      MCMC-style relocation of dead splats, AbsGS-style splitting, and exchange at the cap (free the least
      important splats so growth can continue). This is where Spirula and LichtFeld have both converged. MCMC as
      shipped trails adaptive by about 3 dB on the van even when stable.
- [ ] **Mip-Splatting 3D smoothing filter** (the trainers have only the 2D dilation), and an initial-scale clamp.
- [ ] **Appearance:** evaluate PPISP (physically based per-image correction, in gsplat and 3dgrut) against the
      bilateral grid; keep one. It must not be able to absorb scene colour (Q7), and block runs need one shared
      reference.
- [ ] **Depth and normal priors applied only where views disagree** ("In Depth We Trust", 2026), not everywhere;
      normalise depth error by depth so the weight doesn't depend on scene units.
- [ ] **Pose refinement during training** on by default for capture-pose and video datasets, once the benchmark
      shows a gain (it exists on Linux, switched off; the Mac trainer has none).
- [ ] **Dense initialisation** (in progress in its own session): finish, measure against sparse initialisation
      (EDGS reports reaching 3DGS quality in a quarter of the time), make it the default only on a measured win.
- [ ] **Novel-view consistency** (the sliding-door and haze reports): opacity binarisation late in training and
      contribution-based pruning, judged by the novel-view score.
- [ ] In-run evaluation reports SSIM and LPIPS, not only PSNR.

Done when: every benchmark scene improves or holds on held-out and novel-view scores, with no scene worse by more
than 0.3 dB, and drone held-out PSNR (16–18 dB throughout the history) is above 22 dB or its cause is written down.

### Milestone 4 — Speed (2–3 weeks)

- [ ] **Mac:** remove the two waits per step (`waitUntilCompleted` after preprocess and after Adam): over-allocate
      the tile lists and read the count a step late. Move densification and the colour-grid optimiser to the GPU.
- [ ] **Vulkan:** densification on the GPU (every refine moves all parameters and both Adam moments through the
      CPU today). gsplat: stop running `torch.unique` over all splats each refine.
- [ ] **Schedules:** DashGaussian-style resolution and splat-count scheduling (about 45 % less time reported; it is
      only a scheduler). Then visible-only (sparse) Adam and FastGS-style loss-gated densification.
- [ ] **Data:** stream or cache photos compressed on the Mac instead of holding all of them decoded (about 19 GB
      for 1,250 photos at 2400 px).
- [ ] **Solve:** global mapper first, incremental only when registration is poor (today `auto` runs both up to
      1,000 images); batch LightGlue pairs; keep descriptors on disk (the 9,824-view set needs about 40 GB in
      RAM); essential-matrix verification when intrinsics are known; a learned global descriptor for pair
      retrieval by default.
- [ ] **One memory budget** shared by every stage (decode, extraction, matching, training, upload), replacing the
      five separate guards, with a stress test on a large set and on low RAM.

Done when: each preset's time on the baseline machines drops by at least 30 % with quality held, and no stage can
exceed the memory budget in the stress test.

### Milestone 5 — Networked processing done properly (3–4 weeks)

- [ ] **Block split by visibility**, not only by box: each block gets its cameras plus a ring of outside cameras
      that see into it; outer blocks are clipped to robust scene bounds instead of reaching to infinity.
- [ ] **Merge v2:** drop splats by extent across the cut, not by centre; one shared appearance reference and one
      skybox for all blocks; a short joint fine-tune of the merged scene. Spirula reports 0.1 dB lost on a 3-part
      merge and admits exposure seams — beat that.
- [ ] **Block benchmark:** 1, 2 and 4 blocks on one scene, single device versus several; seam score (held-out
      photos that straddle a cut). Blocks trained on different trainers must hold parity (milestone 2).
- [ ] **Transfers:** one path for every upload — chunked, resumable, checksummed (the worker still takes a single
      POST of up to 20 GB); a test above 16 batches and with a dropped connection.
- [ ] **Worker housekeeping:** free-space check before accepting work, orphan clean-up after cancel (known leak),
      a retention policy for datasets and jobs.
- [ ] **Scheduler soak test:** a scripted mixed queue (process, train, blocks, cancel, pause, migrate) on the RTX
      4080 + Radeon 610M before each Linux release. Slow GPUs take processing and decode, and training only when
      the user chooses.
- [ ] **Transport security** (findings F10, F22, F17, F18, F27, F11): TLS with a certificate pinned at pairing,
      tokens bound to it, `--serve-worker` authenticated.

Done when: a 2-device block run is at least 1.6× faster than one device, within 0.3 dB, with no visible or measured
seam, and the soak test passes three times.

### Milestone 6 — Easy to use (2–3 weeks)

- [ ] **Easy mode is the product:** three presets, no numbers. The app picks splat cap, image size, steps, blocks
      and devices from the scene and the hardware. Learning rates and the other 40-odd raw settings sit behind an
      expert switch, with validation between them.
- [ ] **No silent bad results:** the divergence guard, the colour guard, low photos-placed and a blurry capture
      each stop with one sentence on what happened and what to try.
- [ ] **Capture guidance** (iPhone): per-frame blur score carried into training weights; a motion gate in Stereo
      mode; coverage feedback from the LiDAR mesh for rooms.
- [ ] **GTK parity** in this order: LiDAR depth from capture bundles, Easy mode, the preparation screen, then the
      rest of old phase 2.
- [ ] First run: hardware check, model downloads with sizes and licences, **Try a sample**.
- [ ] Copy diagnostics in every app.

Done when: a new user on each platform gets a good splat from the sample and from their own capture with no
settings changed.

### Milestone 7 — Hardening (old phase 1; can run beside 5 and 6)

- [ ] Re-check all 55 scan findings against `main`; fix every high and medium (untrusted files: PLY / ZIP bounds,
      symlinks, `..`; authentication everywhere; pairing-code limits). Fresh scan afterwards.
- [ ] Fuzz the parsers for a night without a crash.
- [ ] Commercially safe defaults: ALIKED or SIFT + LightGlue instead of SuperPoint; a permissive sky mask; depth
      priors only from Apache-2.0 weights (Depth Anything 3 BASE / SMALL / METRIC-LARGE; not GIANT / LARGE).
- [ ] Written real-hardware passes: Windows + NVIDIA, a discrete AMD card, RTX 50, iPhone, a three-device block
      run.

### Milestone 8 — Distribution and public beta (old phases 3–5)

Signing and notarization (Mac, Windows), a real Flatpak id and Flathub, TestFlight, download pages, capture guide,
troubleshooting, accessibility, localizable strings, then a 10–20 tester beta and 2.0 after two quiet weeks.
Unchanged from the 2026-11-15 plan; the user's decisions in section 7 gate it.

## 5. Issue register

Update this table whenever an issue is touched. "Fixes so far" is there so the next fix replaces them.

| ID | Issue | Evidence | Fixes so far | State |
|---|---|---|---|---|
| Q1 | MCMC training collapses above about 1 M splats | Van capture, block of 322 photos at 2400 px, cap 3.1 M: PSNR 19 → 11 dB near step 8,000 on **both** gsplat and Vulkan (2026-11-17); splats leave the block region, 94 % cut at export. Capped at 1 M: stable at 18–20 dB. Adaptive (430 k): 24–27 dB. | none | **Open.** Test in this order: (a) cap 3.1 M with depth, normal, colour correction and long-axis split off — done 2026-11-17, it collapsed earlier, see below; (b) `noiseLR` 0, then scaled by 1 / extent²; (c) noise gated to zero outside the region and relocation sampling limited to splats inside it; (d) long-axis split off alone (it shrinks the long axis a second time after relocation). Add the divergence guard regardless. |
| Q2 | Low-opacity fog and floaters in exports | Mean opacity 0.12–0.16 on the van; Auto clean's census says invisible splats are "often most of the file" | Auto clean ×2, big-splat penalty, flat-surface work — all judged by eye | Open (milestone 1, 3) |
| Q3 | Presets densify until near the end | Balanced 12 k of 15 k, Preview 4 k of 5 k; High measured better at 14 k of 30 k | — | Open |
| Q4 | Resume restarts the optimiser (Mac); crash loses the run (Linux) | Adam moments zeroed on load; resume state written only on pause | — | Open |
| Q5 | Remote and local solves differ | Different features and engines; stats and log lines dropped | — | Open |
| Q6 | Pose refinement outliers | 17 of 431 cameras differ by 20–28 cm between two solves of one capture | PoseSanity | Open |
| Q7 | Colour correction absorbs scene colour | Drone held-out 14.9 → 9.3 dB; fired on both collapsed van blocks | 4 (off when exposure locked, opt-in, on by default, safeguard) | Open: needs one design |
| Q8 | Block seams and per-block colour | Hard crop by centre; per-block grid mean; sky from one block | none; never measured | Open (milestone 5) |
| Q9 | Camera solve fragility (video, fisheye, drone) | About 12 fix commits; drone places 904 of 1,225 photos | many | Partly solved; needs benchmark numbers |
| Q11 | Off-angle views smear into needles and streaks | Van capture, adaptive, 2026-11-17: clean from the capture direction, broken lettering and streaks from a side angle; earlier sliding-door report | `opacityReg` 0.001 helped one scene, not another; not adopted | **Open — top priority** (milestone 1) |
| Q10 | Capture blur | Earlier limits were undone by ARKit | 5; confirmed at 1/250 s on a phone 2026-11-16 | Solved for LiDAR mode; Stereo mode has no motion gate |
| R1 | Memory exhaustion | 5 incidents incl. two kernel panics | 5 local guards | Open: one budget (milestone 4) |
| R2 | GPU slots, queueing, migration | 7 releases in 3 days; a job queued for ever beside an idle GPU (fixed 1.7.15) | 1.7.3 – 1.7.15 | Works on the RTX 4080 + 610M (migration confirmed 2026-11-17); needs the soak test |
| R3 | Uploads | 400 above 16 batches; single 20 GB POST on the worker | resumable batches, compose fix | Open: one transfer path |
| R4 | Cancelled multi-device run leaves photos on workers | STATUS.md | — | Open |
| R5 | The integrated Radeon 610M is about 60× slower than the RTX 4080 for training | 0.9 steps/s | "slow slot" + migration | Keep it for decode / processing; training only by choice |
| S1 | Security findings | 55 found, about 50 open; F1 high | 5 fixed | Open (milestones 1, 5, 7) |
| P1 | Vulkan matching (`splatwork-vk-match`) | 2.7× OpenCV on a Steam Deck; matching is seconds per capture | stage 1 shipped | Closed: stage 2 not worth it |

**Q1 test (a), 2026-11-17 (job `j71f8a9b1859`, Vulkan trainer, RTX 4080):** with depth, normal, colour correction
and long-axis split all off, MCMC collapsed **earlier** — 16.0 dB at step 4,000, 12.1 dB at step 5,500 with only
416 k splats, 13.1 dB at step 10,000. So those terms are not the cause; depth supervision was holding the run
together. The fault is in the MCMC core on this scene (position noise, relocation, opacity regulariser, or splats
leaving the region; scene extent 5.1). Tests (b) and (c) are next; (d) is ruled out as the main cause.

## 6. Research to adopt, and when

| Area | Adopt | Milestone | Notes |
|---|---|---|---|
| Densification | Hybrid: error-guided growth + relocation + AbsGS split + exchange at cap | 3 | Both GPL competitors converged here; implement from papers |
| Anti-aliasing | Mip-Splatting 3D filter | 3 | Low cost |
| Appearance | PPISP vs bilateral grid | 3 | Apache-2.0 references in gsplat / 3dgrut |
| Geometry | Depth / normal priors only on inconsistent regions; PGSR-style multi-view planar consistency later | 3, later | Mesh export depends on this |
| Initialisation | Dense matcher / MVS / depth-fusion init | 3 | In progress |
| Speed | DashGaussian scheduling, sparse Adam, FastGS loss-gated growth | 4 | FastGS claims about 100 s per scene on a 4090-class card |
| Solve | Global mapper first; feed-forward poses (VGGT / π3 class) as a fallback when SfM fails, then bundle adjustment | 4, later | Check weights' licences; several are non-commercial |
| Large scenes | Visibility partition, ownership merge, shared appearance; LoD / out-of-core later | 5, later | The networked pillar |
| Lenses | 3DGUT (fisheye, 360°, rolling shutter without undistortion) | after 6 | Now standard in Spirula, LichtFeld, gsplat; a new projection path in each trainer |
| Delivery | SPZ v4, SOG, glTF `KHR_gaussian_splatting` (check ratification), HTML export | after 6 | Low cost; users value it |
| Mesh | Splat-to-mesh export | after 6 | The most-valued competitor feature SplatWork lacks |
| 4D / feed-forward splats | Watch only | — | Not mature for a best-quality product |

## 7. Scope: what is core

| Tier | What | Rule |
|---|---|---|
| Core | Mac app (Metal trainer, viewer, prep); Linux worker with gsplat on NVIDIA; splatprep photo / video / iPhone-LiDAR solves; Mac ↔ worker offload and block training | Every milestone serves these first |
| Secondary | Linux GTK studio; drone / GPS; 360° video; editing and Auto clean; SplatWork Capture LiDAR mode | Kept working; improved in milestones 3–6 |
| Experimental (frozen until milestone 2 is done) | Windows app; Vulkan trainer and AMD / Intel runtimes; home video (VHS); stereo and fisheye sources; Capture Stereo mode; `splatwork-vk-match`; Steam Frame | Off by default or labelled; bug fixes only |

After milestone 6, in this order: 3DGUT lenses, mesh export, SPZ / SOG / glTF export, Steam Frame capture,
multi-source datasets, 4D.

## 8. Decisions for the user

1. **Scope freeze:** agreed by the user on 2026-11-17 — Windows, the Vulkan trainer, VHS and stereo get bug fixes
   only until milestones 0–2 are done. The user's top priorities are Q1 (MCMC) and Q11 (off-angle views).
2. **One strategy:** replace the MCMC / adaptive choice with a single hybrid strategy (milestone 3), keeping the
   old ones only behind the expert switch.
3. **Commercial use by default** (permissive models), or best models by default with licences shown.
4. Apple Developer Program, Windows signing and a new Flatpak id (milestone 8), as in the earlier plan.
5. Who it is for: people who download apps, or people who build from source. This sizes milestones 6 and 8.

## 9. Sources for this review

Five read-only reviews on 2026-11-17 (Mac app, Linux worker and trainers, prep and capture, project history,
research survey), the diagnostic jobs on the RTX 4080 the same day, CHANGELOG.md, STATUS.md and the security scan
in `CLAUDE-SECURITY-20261109-064902/`. Research items marked with claims are from the papers' own numbers and
were not reproduced here.
