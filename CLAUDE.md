# SplatWork: instructions for development sessions

Read [docs/dev/ROADMAP.md](docs/dev/ROADMAP.md) first. It is the plan: its **Rules** (section 3) apply to every
change, work comes from the **first unfinished milestone**, and the **Issue register** (section 5) lists every
recurring problem with the fixes already tried. [docs/dev/STATUS.md](docs/dev/STATUS.md) says what is tested and
how releases are made.

## The short version of the rules

1. **Measure before and after.** A change to training, solving, defaults or presets runs the benchmark
   (`tools/eval/`, ROADMAP milestone 0) and puts the before / after numbers in the commit message. Never judge
   splat quality by eye alone.
2. **One default change per commit.** No same-day flip and revert. Unmeasured options are measured or removed.
3. **Experimental means off by default** until a written real-hardware run exists in `docs/dev/`.
4. **Scope freeze until milestones 0–2 are done:** no new platforms, sources or trainers. Put ideas in ROADMAP
   section 8 instead.
5. **Three trainers, one specification** (Metal `mac/Sources/SplatStudio/Train`, gsplat
   `linux/splatworker/trainer.py`, Vulkan `linux/vktrain`): a training change lands in all three or is marked
   single-trainer.
6. **Fix the class, not the instance:** check the Issue register for earlier fixes of the same problem first, and
   update the register in the same commit.
7. Batch Linux releases; `main` must build every target, including the iPhone app.

## Standing project rules

- Push every commit to GitHub.
- Never quit or replace the running Mac app: install builds with `mac/scripts/update_local.sh`.
- These identifiers never change: `com.example.SplatWorker`, `_splatworker._tcp`, `_splatcapture._tcp`,
  `SplatStudio`, `SplatStudio.WorkerToken`.
- Names: SplatWork (Mac), SplatWork for Linux, SplatWork for Windows, SplatWork Capture (iPhone).
- Licences: Spirula Studio and LichtFeld Studio are GPL-3.0, and most paper code derives from the non-commercial
  Inria code. Take ideas from papers; use only Apache-2.0 code (gsplat, NVIDIA 3dgrut) as a reference.
- Never commit footage or personal media (`Sample datasets/` is git-ignored).
- Don't stop the user's processes or submit jobs to their PC while their own jobs run without asking.

## Tests

- Linux: `cd linux && python -m pytest -q tests`
- splatprep: `cd tools && python -m pytest -q tests`
- Mac: `cd mac && swift build -c release` (a `swift test` target is milestone 2)
- Vulkan trainer: `linux/vktrain/ci/Dockerfile` (Mesa lavapipe)
