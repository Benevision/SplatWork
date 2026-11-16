# Changelog

Notable changes to SplatWork. The Mac app is released by version tag (see
[Releases](https://github.com/Benevision/SplatWork/releases)); a rolling *developer build* follows `main`.
SplatWork for Linux is published to its Flatpak repository on every change and carries its own version number.

## Unreleased

- **SplatWork for Linux 1.7.12 — the whole run as one job on the worker:** when a project's processing and its
  training go to the same worker, the Studio now does what the Mac does: the photos (or the video) and every
  setting go up once, and that worker runs frame extraction, masks, the sky, the camera solve and training on its
  own. This PC can go offline meanwhile, for up to a day. When training starts there it fetches the prepared
  dataset, then the previews and the result. *Prepare again* with the same settings picks the running job up
  again instead of starting over. When that isn't possible (several stacked mask models, a home video, a worker
  older than 1.6, a missing model), the log says why and the steps run one by one as in 1.7.11.
- **SplatWork for Linux 1.7.11 — Process on another worker, as on the Mac:** *New Project → Processing device →
  Process on* offers *This PC* or any worker. Workers that can't process or are offline are marked, and the choice is
  remembered. Video splitting, masks, the sky and camera solving then run on that worker: the photos (or the video,
  with *Split videos on the worker*) go up once and stay there, so later steps, and training on that worker, don't
  send them again. Only the masks and settings a step needs go up, and only its output comes back. Steps the worker
  can't do run on this PC: a model it doesn't have (the dialog says which), or an error while splitting or masking.
  A worker that doesn't answer within 20 s means this PC. The project shows "processed on …". Block-parallel
  training from the Studio now gives a Linux worker on 1.7.8 or newer one block per training GPU, as the Mac does.
- **Fixed: preparing a large photo set could run the Mac out of memory and crash it.** COLMAP's feature extraction
  (used for photos, and to refine iPhone captures) ran on every core, and each thread needs about half a gigabyte for
  a 1600 px photo, more for bigger ones. On 431 high-resolution iPhone stills it reached 15 GB, and a 24 GB Mac ran
  out of memory and swap and restarted (kernel panic). Three changes:
  - COLMAP now gets photos at the size it extracts at (results are scaled back to full size).
  - Its thread count is capped at about a fifth of the computer's memory.
  - Every tool splatprep runs is stopped with a clear error if it passes 60 % of the memory, rather than taking the
    system down.

  The same 431 photos now peak at 4.4 GB.
  Feature extraction shows its progress again ("extracted 377/431"): COLMAP 4 prefixes its log lines, and
  splatprep didn't recognise them, so the bar stood still for the whole step. The thread cap allows 30 % of the
  memory (5 threads for 2400 px photos on a 24 GB Mac, about 6 GB) instead of 20 % (3 threads).

- **The crop box can be turned.** *Edit → Crop box → Rotation*: X, Y and Z sliders turn the box about its centre, for
  scenes that don't line up with the axes (a van parked at an angle). *Fit to scene* fits the box along its own turned
  axes, and *Reset rotation* squares it up again. Hiding, *Apply crop* and the box outline all follow the rotation.

- **Several projects can prepare at once.** *New* while a project is preparing starts another one alongside it, for
  example on this Mac while the first uploads to a worker. The preparation page also has a *New Project…* button.
  Each running preparation gets its own progress button in the toolbar. Each project page shows and stops its own
  run, and Resume no longer waits for another project. A project trains with the settings it was started with. One
  that finishes while another project is training waits as *Ready to train*. One finishing in the background no
  longer closes the window you're working in.

- **SplatWork Capture: sharper frames.** The shutter limit (default 1/250 s; now also 1/1000 s) is enforced by the
  app in both modes: a few times a second it sets the shutter itself, never slower than the limit, and steers ISO
  by the camera's metering. A locked recording holds the last values. Before, the limit was applied once when
  exposure locked, and ARKit undid it (captures at 1/91–1/100 s). The camera's built-in limit was ignored too, and
  Stereo mode had none at all (1/33–1/50 s indoors). The dual camera used by Stereo mode doesn't allow custom
  exposure, but there the camera's own limit holds. Checked on an iPhone: a LiDAR capture at 1/250 s, ISO 1600, and
  a Stereo capture at 1/250 s, ISO 1000–6400. A shutter limit saved as "Auto" (no limit) moves once to 1/250 s.
  Both modes log the applied exposure to the Xcode console. New *Torch while recording* for dim rooms. *High-resolution stills* is now on by default
  for new installs: in a side-by-side test they were far less blurred than video frames. Stereo mode falls back
  without the depth stream on iPhones that can't stream depth while taking photos from both cameras (they stopped
  the camera: a frozen preview).

- **SplatWork Capture: Stereo mode (new, not yet tried on a phone).** A *LiDAR / Stereo* switch on the camera screen.
  Stereo takes photo pairs from the wide and ultra-wide cameras at once (iPhone 11 or newer), each with the lenses'
  factory calibration and, where the phone supports it, the dual camera's depth map. SplatWork on the Mac solves the
  cameras like photos and keeps each pair at the lenses' calibrated offset, so the scene comes out in metres. The
  ultra-wide's wider view gives the solve more to match. Checked on the Mac with a synthetic pair set: the rig was
  used and the 12 mm offset came out exactly; not yet recorded on a real iPhone.
- **SplatWork Capture builds again.** Two files the shared code started using in 1.3.0 (home-video analysis and the
  batched uploads) weren't in the iPhone project, so building it failed with "Cannot find 'HomeVideoAnalysis' in
  scope".

## 1.3.0 — 2026-11-16

SplatWork 1.3.0 for Mac, SplatWork for Linux 1.7.10, and the first SplatWork for Windows. SplatWork Capture is
unchanged (1.2).

### Highlights
- **SplatWork for Windows (new):** the Linux app's worker and studio for Windows 10 / 11 PCs with an NVIDIA GPU,
  RTX 50 cards included, as an MSI installer or a portable zip. Tested in CI only so far, not yet on a real PC.
- **AMD and Intel graphics on Linux (experimental):** recognised by the app, with ROCm / XPU processing runtimes and
  a new Vulkan trainer. On Linux every GPU can train its own job, integrated ones included. A slower GPU starts
  waiting jobs, which move to a faster one when it frees up. *Measure GPU speed* and *Compare trainers* (CUDA vs
  Vulkan) are built in.
- **Home video (VHS, camcorder tapes):** restores old tape footage (deinterlacing, noise, dropouts, colour) and picks
  the clip best suited to a splat.
- **Drone surveys that hold together:** GPS-guided matching, a re-solve held to GPS when the fast solve folds the
  scene or places too few photos, take-off shots left out, and a GPS check before training.
- **A new preparation screen on the Mac** with a step timeline, live counters and previews of masks, sky and
  cameras. Uploads to workers are smaller, batched and resumable.
- **Safer colour correction:** training notices when per-photo colour correction takes over the scene's colour and
  resets it.
- **SplatWork for Linux checks for and installs updates itself.** Its Flatpak repository is now signed: see the
  install notes in linux/README.md.

### Details

- **Block runs use every GPU on a Linux worker.** A worker with several training GPUs (SplatWork for Linux 1.7.8 or
  newer, e.g. an RTX 4080 with an integrated Radeon) now gets as many blocks at once as it has GPUs. Before, the Mac
  sent it one block at a time. The worker decides where each block runs and moves a block to a faster GPU when one
  frees up. Each block's row in *Devices* and the progress bar show the GPU it is on ("BensPC · RTX 4080").
- **SplatWork for Linux 1.7.10 — video decoding on the NVIDIA GPU's video decoder:** frame extraction and home-video
  restoration can decode H.264, HEVC and AV1 on NVDEC, a separate part of the GPU, so the CPU stays free and
  training barely notices. It is used only when it gives the same pictures as the CPU: the first frames of each
  video are decoded both ways and must match (grey images identical, colour within 2 levels). Otherwise, and on any
  error, decoding falls back to the CPU. *Settings → Training → Decode video on: Automatic / CPU*. The job log says
  which decoder ran and at how many frames per second. AMD and Intel GPUs keep decoding on the CPU: PyAV's FFmpeg
  has no VA-API.
- **SplatWork for Linux 1.7.9 — compare the CUDA and Vulkan trainers in the app:** *Models & Benchmarks → Compare
  trainers* (or `splatworker benchmark --compare-trainers [--quick] [--both-strategies]`) trains the benchmark
  scene on the same GPU with gsplat and then with the Vulkan trainer, at the same settings and seed. It shows a
  table with speed, total time, peak GPU memory, splats, held-out PSNR and SSIM, and a summary line such as
  "Vulkan: 0.42× the speed, −0.3 dB". Results are kept in `trainer-comparisons.json` with the job logs. Both
  trainers now report held-out SSIM, computed the same way. A training job can ask for a trainer
  (`"trainer"` in `POST /v1/jobs`).
- **SplatWork for Linux 1.7.8 — a slower GPU starts waiting jobs, which move to the faster GPU when it frees
  up:** on a PC with a fast and a slow GPU (an RTX 4080 and the Ryzen's Radeon 610M), a training job waiting for
  the busy fast GPU, block-run jobs included, now starts on the slower one if it fits its memory. When the fast GPU
  frees up, the job pauses, its state is saved, and it continues there from the same step under the same job.
  The log says "moved from Radeon 610M to RTX 4080 at step N". A job that is almost done stays where it is.
  Moving needs the Vulkan trainer on both GPUs. *Settings → GPUs* gives each slower GPU a choice: *Only when
  faster GPUs are busy* (default) or *Always use, like any other GPU*, with its measured share of the fastest GPU.
  A switch, *Move jobs to a faster GPU when one is free*, can turn the moves off. `/v1/info` "slots" adds `slower`
  and `userChoice`, and JobStatus adds `slot`.
- **SplatWork for Linux 1.7.7 — Check for Updates…:** in the main menu and in *Settings → Startup and background*.
  It checks right away and says whether this version is up to date, that a newer one is available, or why it
  couldn't check. In the Flatpak, *Update and Restart* downloads and installs it straight away, even before the
  automatic check has noticed it, then restarts. With jobs running or queued it offers *Update After Jobs*
  (download now, restart once they have finished) or *Stop Jobs and Update*. A source build shows the download page
  instead, and SplatWork for Windows uses its own installer.
- **SplatWork for Linux 1.7.6 — processing uses the GPU only for its GPU steps:** a camera solve, masks or frame
  extraction used to hold a GPU for its whole run, so it waited behind training even for CPU work such as COLMAP
  mapping. Processing jobs now run on the CPU (one at a time by default: *Settings → Training → Processing jobs at
  once*) and take the GPU only for feature extraction, matching, masks and 360° resampling. A GPU step waits for
  the GPU, or runs beside training when the GPU has enough free memory. Queued training comes first. The Jobs page
  shows "running on the CPU" or "waiting for the GPU", and running processing jobs now show their progress and can
  be cancelled there.
- **SplatWork for Linux 1.7.5 — GPU speed measured under real load:** *Measure GPU speed* trained a tiny scene
  for about a second on an RTX 4080, too little to load a big GPU. It now trains the standard benchmark scene
  (48 photos of 1600×1200, 1 M splats) and times 35 s after a 10 s warm-up, about a minute per GPU, with progress
  shown per GPU. A GPU the scene doesn't fit falls back to a smaller scene, and speeds are compared only at the same
  size. Peak GPU memory is recorded too. `/v1/info` lists every Vulkan GPU's measured speed (`"vulkanDevices"`),
  ticked or not. Earlier measurements count as unmeasured: measure again.
- **SplatWork for Linux 1.7.4 — GPU list fixes:** *Settings → GPUs* listed the RTX 4080 twice and an integrated
  Radeon twice (named after the CPU, ticked) on a PC where the Flatpak sees every GPU through two driver copies.
  Each GPU is now listed once (same PCI address = one GPU, the newest driver copy kept), integrated GPUs are
  recognised from the system's PCI scan and named as such ("AMD Radeon 610M (integrated)") and start unticked, and
  a CUDA GPU is matched to its Vulkan device by PCI address, name, or vendor and memory size. *Measure GPU speed*
  measures each GPU once.
- **SplatWork for Linux 1.7.3 — every GPU trains with the Vulkan trainer:** with the Vulkan trainer chosen, each
  Vulkan GPU is its own job slot, so a PC with two cards trains two jobs at once. *Settings → GPUs* now lists GPUs
  only the Vulkan trainer can use (AMD, Intel, integrated). An integrated GPU next to a faster card starts unticked
  and, when ticked, only takes a training job of its own while the faster GPUs are busy, never a block of a block
  run, so existing runs are as fast as before. Speeds are measured rather than guessed: *Settings → GPUs → Measure
  GPU speed* (or `splatworker measure-gpus`) trains a small scene on each GPU, and a GPU under a quarter of the
  fastest counts as slow; an unmeasured integrated GPU counts as slow. `/v1/info` lists the slots with their
  measured speeds (`"slots"`); `splatworker benchmark --slot` tests one GPU. Processing and Auto clean stay on CUDA
  GPUs.
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
- **Drone surveys: a fast solve that places too few photos is solved again, held to GPS.** When the quick (GLOMAP)
  solve matches GPS but places under 70 % of the photos, the cameras are solved again held to their GPS positions.
  The new solve is kept if it also matches GPS and places more. On the 1,225-photo survey: 498 placed, then 904,
  0.95 m from GPS.
- **Training on a Linux worker waits for it.** When a run should start on a worker that's briefly unreachable (for
  example updating itself), the Mac waits up to 15 minutes ("Waiting for BensPC — it may be restarting after an
  update") instead of failing with "No Linux worker selected".
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
  - RTX 50-series cards get the same PyTorch 2.7 + CUDA 12.8 runtime as on Linux. SplatWork's CI compiles gsplat 1.6
    for Windows for it. This is untested on a real RTX 50 card.
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
