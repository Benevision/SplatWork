# Changelog

Notable changes to SplatWork. The Mac app is released by version tag (see
[Releases](https://github.com/Benevision/SplatWork/releases)); a rolling *developer build* follows `main`.
SplatWork for Linux is published to its Flatpak repository on every change and carries its own version number.

## Unreleased

- **Smaller, resumable uploads to workers.** Photos sent to a Linux worker are re-saved for the upload only; the
  project keeps the originals. *Settings → Uploads to workers → Upload size* offers three options:
  - **Standard** (default): about ¾ the bytes, 38 dB from the original;
  - **Compact:** about half, 36 dB;
  - **Original.**

  Both re-saved sizes differ from the originals far less than a splat can show. Location and a drone's height
  above take-off are kept. Uploads go in batches of about 200 MB that the worker combines (every worker since
  1.3.8), and the Mac remembers which batches arrived. A retry after an interruption sends only the missing ones,
  where before it started again from zero. Videos are sent as they are. On 300 drone photos (1.3 GB): Standard
  sent 1.0 GB, Compact 740 MB, and a stopped upload resumed from its last finished batch.
  Uploads of more than 16 batches (about 3 GB) failed with "Worker error 400: invalid dataset … image(s)
  missing". When the batches were combined in groups, the group holding dataset.json didn't have all its photos.
  Fixed on both sides: the Mac now adds dataset.json only in the final combine, and the worker (SplatWork for
  Linux 1.7.2) keeps such a group as a partial layer instead of refusing it.
- **Drone surveys: take-off shots left out, and a few stray cameras no longer fail the run.** Photos taken near
  the ground while the rest of the flight is well above (DJI's height above take-off: under 5 m, or under 30 % of the
  flight's usual height) are left out before the camera solve. The new switch *Leave out take-off and landing shots*
  is on by default. After the solve, when the cameras as a whole match GPS (median error under a quarter of the
  limit), up to 15 % that are far from their GPS position are left out instead of stopping the run. On the
  1,227-photo survey a good solve, typically 1 m from GPS with 4 % far off, used to stop. A folded solve still stops.
- **A new preparation screen on the Mac:** a progress ring with elapsed time and time left. A timeline shows every
  step with its time: photos, upload, masks, sky, camera solve, finalise, download. A card for the current step
  says where it runs (this Mac or a Linux worker) and gives live counters with a rate ("9,866 of 14,128 pairs"),
  the camera-solve sub-steps and the recent activity. Live previews show the latest photos, the masks and the sky
  tinted onto the photos, and a top-down map of the solved cameras over the sparse points. Uploads to a worker
  show size, speed and time left. The full log is one click away.
- **The same colour-correction safeguard on the Mac's trainer.** On healthy runs it stays quiet (checked on an
  iPhone capture and a drone survey).
- **SplatWork for Linux 1.7.1 — colour-correction safeguard:** per-photo local colour correction (bilateral grid)
  can no longer take over a scene's colour, as it did on a drone survey trained on a wrong camera solve (the result
  came out nearly black, its colours held by the correction, while held-out PSNR fell from 14.9 to 9.3 dB). From
  step 4000, at each eval, both Linux trainers (gsplat and Vulkan) score 8 training photos rendered without the
  correction. Two evals in a row more than 3 dB below their best, or renders under 0.6 of the photos' brightness,
  reset every grid; the second time, the correction is turned off for the rest of the run, which a resumed job
  remembers. The job log and status say so and point at the camera solve. Same check as the Mac trainer's.
- **SplatWork for Linux 1.7.0 — training on AMD and Intel graphics cards (experimental):** a new Vulkan trainer is
  built into the app. It needs no PyTorch or CUDA download and runs on any GPU with a Vulkan driver.
  - On AMD and Intel cards it trains by default; NVIDIA cards keep the gsplat (CUDA) trainer, and the Vulkan one can
    be chosen in *Settings → Training → Trainer*.
  - It reads the same training settings as the CUDA trainer: both densification strategies, skybox, appearance
    correction and bilateral grid, depth and normal losses, moving-object masking, pose refinement, the compact
    optimizer, previews, checkpoints and pause / resume. Depth priors are predicted with PyTorch when the processing
    runtime is installed.
  - The setup assistant says the card trains with the Vulkan trainer, and the Mac sees the card and the trainer
    (`/v1/info` "trainer").
  - Not yet run on a real GPU: it has only been checked against gsplat on Mesa's software Vulkan driver. Quality and
    speed on real cards are still to be measured (`linux/vktrain/compare_training.sh`). Densification still copies
    the splats to the CPU and back, and Auto clean still needs gsplat.

- **Drone surveys no longer fold or double the scene:** photos with GPS are now matched with their GPS
  neighbours (DJI's numbered file names made them look like video, so neighbouring flight lines were only linked
  by look-alike guesses — identical sheds matched to each other). If the solve still disagrees with GPS, it is
  solved again held to the GPS positions, and the parts of that solve (e.g. two flights at different heights)
  are combined. On a 1,227-photo property survey the cameras went from a median 48 m off GPS (scene doubled) to
  0.96 m. If a solve still doesn't match GPS, preparation stops with a clear message instead of training on it
  (turn off *Use photo locations* to keep it anyway).
- **Home video (VHS, camcorder tapes), new:** a *Home video* tile on the Mac and in SplatWork for Linux turns
  digitised old tapes into splats.
  - The video is read once and split into clips at scene cuts. Each clip gets a verdict: the camera moves; it only
    turns; it hardly moves; it's blurred; it's mostly moving people, water or a car; it's too short; or the picture
    is blank. The clip with the most depth is ticked; tick others of the same place to join them.
  - Frames are picked where the camera has moved on and the picture is sharpest. Repeated frames from tape transfers
    are skipped.
  - Each frame is cleaned: black borders and the head-switching band are cropped, pixels are made square, interlaced
    frames are rebuilt, tracking lines are repaired, the frame is denoised with its neighbours, and each clip's colour
    cast is taken out. Burnt-in date stamps are masked.
  - Each clip, and each zoom setting within a clip, gets its own camera. A ticked clip that the solve can't place
    with the others (its cameras land on one spot, or it gets an impossible zoom) is left out, with the reason.
  - Training uses SH degree 1 and ignores moving people.
  - It runs on the Mac or on a Linux worker. Results and limits are in `docs/dev/VHS.md`.
- **Video with non-square pixels is no longer stretched:** PAL / NTSC captures and re-encoded tape transfers store
  pixels that aren't square. Their frames are now resampled to square pixels on the Mac and on Linux workers, so the
  camera solve isn't given a squashed picture.
- **SplatWork for Windows (new, untested on real hardware):** the Linux worker and studio, with the same GTK 4
  interface, packaged for Windows 10 / 11 as an `.msi` installer and a portable `.zip`.
  - It includes the Projects list, the new project flow, the viewer and editor, Compute devices and the setup
    assistant. The assistant installs PyTorch, CUDA and gsplat for NVIDIA cards using gsplat's Windows wheels.
  - Macs find the PC over Bonjour. The installer opens the worker's port and mDNS on Private networks only.
  - It updates itself from new releases, checking each installer against the release's checksums.
  - Built, installed and smoke-tested on GitHub's Windows runners, which have no GPU. Training on an NVIDIA GPU,
    the viewer on real drivers and discovery from a Mac over a real network haven't been tried yet.
  - AMD and Intel GPUs can view and process; they'll train once the Vulkan trainer lands.
- **SplatWork for Linux 1.6.0 — AMD and Intel graphics cards:** the app now recognises AMD Radeon / Instinct and
  Intel Arc cards (name, video memory, load), shows them on the Status page and to the Mac, and explains that
  training still needs an NVIDIA GPU instead of saying "no GPU found". On those cards it offers a processing runtime
  with PyTorch for ROCm (AMD) or for Intel GPUs, so feature matching and masks can use the GPU. Not yet tested on
  real AMD or Intel hardware; training on AMD / Intel is not available (options and plan in
  `docs/dev/GPU_SUPPORT.md`).
- **Whole pipeline on one worker:** when a project's processing and training both go to the same SplatWork for
  Linux worker (1.6.0 or later), the Mac uploads the video (or the photos it imported) and every setting once, and
  the worker runs frames, masks, sky, camera solving, levelling and training by itself — the Mac can sleep, go
  offline or quit meanwhile. Reopening the app (or *Reconnect to the Worker Job* on the project page) picks the job
  up and fetches the dataset, with the same per-stage checkpoints a local run leaves, and the result. Older workers,
  Apple Vision masks, Object Capture and stereo or posed sources keep the step-by-step flow.

## 1.2.0 — 2026-11-13

SplatWork 1.2.0 for Mac, SplatWork for Linux 1.5.0.

### Highlights
- **Never start over:** every processing stage is checkpointed and training saves checkpoints and finished blocks,
  so quitting, a crash or Stop costs at most the step that was running. A new **Projects** list shows where every
  project stands and resumes or redoes it from any stage.
- **Drone footage and photo locations:** imported photos keep their GPS (only that), so camera solving checks the
  cameras against it and puts the scene in metres. Drone photos are recognised by the drone's own markers — not by
  GPS, so phone photos stay ordinary — and get aerial settings; there's a new **Drone / aerial** tile.
- **Training on several devices, sturdier:** Retry or leave out a failed block without stopping the rest, Pause and
  Stop reach every device, a live view of block runs, and Show Live View to get back to a run from anywhere.
- **SplatWork for Linux:** the Linux app (formerly SplatWork Node) now has the Mac's layout and wording, and keeps
  training jobs' saved state when it is stopped.
- **New look:** the hex mesh logo on every app, the website and the docs.

### Projects and resuming
- **Projects list** in the sidebar replaces Recent: every project you've worked on, including ones made with New
  Project and training results, with its status (processing, interrupted, ready, training, trained).
- **Project page**: each processing stage with what it produced, *Resume*, *Redo from Here*, *Change Settings…*,
  *Train*, and every result. The start page offers projects that stopped partway.
- **Processing checkpoints**: each stage (frames, masks, camera positions, dataset) is saved and reused on any rerun
  whose settings and source files haven't changed. Quitting, a crash, Stop or a failure costs at most the stage
  that was running. *Try Again* after a failure continues where it failed.
- **Training resume**: this Mac checkpoints every 10 minutes and on Stop; block runs keep every finished block and
  reuse it.
- **Choose the processing device** in New Project (Cameras & masks page) and when resuming, separately from the
  training devices; the training devices you tick are remembered.

### Photo locations and drone footage
- Imported photos keep only their location (GPS, and a drone's height above take-off), on the Mac and on Linux,
  under a new **Use photo locations** switch; off strips locations from the project's photos.
- Drone footage (camera maker, DJI flight / gimbal tags, DJI file names) gets **Aerial settings**: learned camera
  solving with GPS checks, the sky as a learned skybox, the High quality preset.
- The Maximum preset stores photos at its 2400 px training resolution.
- Sky masking is on for video, photos and 360° whichever way the project is started, and is skipped with a note
  (not a failed run) when no sky model is available.

### Training on several devices
- A failed block waits for **Retry** (on any device) or **Leave This Block Out** instead of failing the whole run.
- **Pause / Resume** now reach worker jobs and blocks, not only a single run on the Mac. **Stop** stops a block
  training on the Mac too, and shows *Stopping…* while the run winds down.
- **Live view for block runs**: the viewer shows every block's latest state, merged.
- **Show Live View**: look at other projects while one trains, then get back to it from the toolbar, the banner,
  the Train tab or the project page. Home no longer stops training.
- Charts no longer draw a line back to zero when a block restarts; block cards keep the device's name if it drops
  off the network.

### Quality
- **Flat surfaces**: surface-normal consistency loss and flat-splat regulariser on by default; predicted depth
  (Depth Anything V2 on the Mac, monocular depth priors on Linux) for photos without a depth sensor.
- **Fisheye footage**: solved with a fisheye lens model, with masks warped along.
- Camera solving drops cameras placed far off the camera path.

### SplatWork for Linux 1.4–1.5
- The Linux app is now a full studio: viewer, New Project, editing (including Auto clean and the mask editor),
  export, turntable videos, iPhone Inbox, compute devices and block-parallel training across machines, all next to
  the worker server.
- Renamed from SplatWork Node to SplatWork; sidebar, start page and New Project follow the Mac's.
- Worker jobs can be paused (state saved, GPU freed) and checkpointed; stopping the worker pauses running training
  jobs instead of discarding their state.
- Multi-device runs follow a device that changes address and retry a block here after 30 minutes unreachable.
- The setup screen shows the exact command to install a missing NVIDIA driver extension.

### Fixes
- About 30 issues from an independent review of this release's new code, among them: masks from an earlier run
  baked into reused frames; two preparations running in one folder after Stop; Stop not reaching a block
  training on the Mac; a failed block holding up its device's other blocks; resumed runs on the wrong devices or
  from step 0; the Linux mask editor not showing why a job failed.

### Security
- Mac worker requests need an access token; Bonjour-advertised addresses are only trusted once the worker confirms
  them; dataset paths can't escape their folder; benchmark runs are isolated.

### Changes
- Licensed under Apache 2.0; public documentation, website and project files.
- The repository moved to `Benevision/SplatWork`.
- Local Mac builds are signed with a development identity when there is one, so the Keychain stops asking again
  after every update.
- iPhones no longer act as training workers for other devices; SplatWork Capture still processes and trains its own
  scans.

## 1.1.0 — 2026-11-08

- Self-updating apps: the Mac app updates from GitHub Releases; SplatWork Node updates itself through Flatpak.
- A developer-build update channel on the Mac (Settings → Developer).
- The iPhone app makes splats on its own and can hand the heavy steps to Macs and PCs; a Mac can act as a worker.
- Compute devices are checkboxes: this Mac and each worker.
- A job too big for its one device trains there in blocks that fit instead of being refused.
- Learned skybox on the Mac trainer, saved alongside the PLY.
- The preparation window can be hidden while it works.
- Renamed to SplatWork (Mac), SplatWork Capture (iPhone) and SplatWork Node (Linux).

## 1.0.0 — 2026-11-08

First release, as *Splat Studio*: Metal viewer and trainer, dataset preparation from photos, video, 360° video,
stereo footage, iPhone captures, COLMAP and Nerfstudio, Linux CUDA worker, editing and export.
