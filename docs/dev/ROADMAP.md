# Roadmap: from a working suite to a public release

Written 2026-11-15 from `main` at `8c13b7f` (see [STATUS.md](STATUS.md) for what is tested and known issues). The
goal: someone who isn't us downloads SplatWork, installs it without warnings, gets a good splat on supported
hardware the first time, isn't exposed to network or licensing risk, and can tell us when something goes wrong.

## Where it stands

The feature set is broad: five apps (Mac, Linux, Windows, iPhone, splatprep) and every capture source on the
original roadmap, including drone surveys and home video. The gaps are in hardening, real-hardware testing and
distribution, not in features.

| Area | State | Gap for wider use |
|---|---|---|
| Mac app | Main app, used daily, end-to-end on real data | Ad-hoc signed (Gatekeeper warns); no automated Swift tests; 2 accessibility labels; English only |
| SplatWork for Linux | Used on the RTX 4080 PC; 362 tests | Flatpak id `com.example.SplatWorker` (placeholder; blocks Flathub); behind the Mac in the interface |
| Vulkan trainer (AMD / Intel) | Stages 1–3, default on AMD / Intel since 1.7.0 | Never run on a real GPU; densification on the CPU |
| SplatWork for Windows | MSI + zip, CI on `windows-latest` | Never run on a real PC or GPU; unsigned (SmartScreen); RTX 50 wheel still building |
| SplatWork Capture (iPhone) | 1.2, builds | Build from source only; not run on a phone for 1.2 |
| Security | 5 of 25 review findings fixed (F2, F3, F5, F6, F21) | About 20 open, incl. plain HTTP with bearer tokens, the unauthenticated iPhone worker, pairing-code brute force, out-of-bounds reads in the PLY / ZIP parsers, symlinks in worker outputs |
| Licensing | Model licences shown before download | The **defaults** use non-commercial weights: SuperPoint (feature matching) and SegFormer (sky masks) |
| Support | Bug / feature issue templates, SECURITY.md | No way for a user to gather logs and system details for a report |

## Phase 0: finish what's in flight (days)

1. **Drone fixes** (code written; a check solve of the 1,229 photos is running): GPS-held re-solve when the fast
   solve places < 70 %; worker-bound training waits up to 15 min for an unreachable worker. Commit, then Redo from
   Cameras on the Drone project.
2. **RTX 50 wheel for Windows** (Windows session): land it, merge `claude/windows-app-84ce39`.
3. **Real-GPU checks of the Vulkan trainer:** RTX 4080 comparison (`compare_training.sh`), RX 6700 XT on SteamOS
   (`AMD_TEST_STEAMOS.md`). If it is not close to gsplat, make it opt-in on AMD / Intel too and label it
   experimental.

Done when: all three are in, or written down as known limits.

## Phase 1: hardening (2–3 weeks)

Nothing goes to outside users before this phase is done.

1. **Security findings.** Re-check each open finding against `main` (some code has moved since 2026-11-09), then fix
   them in groups:
   - *Transport:* HTTPS between the apps with a self-signed certificate per device, pinned at pairing (trust on
     first use), so tokens and datasets are no longer cleartext (F10, F22). Bind saved tokens to that certificate
     instead of the spoofable Bonjour name (F17, F18, F20).
   - *Authentication everywhere:* iPhone worker (F14, F15, F16), `--serve-worker` (F11), the Mac's private datasets
     (F9). Pairing codes: attempt limit, lockout, rotation (F7).
   - *Untrusted files:* refuse symlinks and `..` when unpacking worker outputs and masks (F1, F25); range-check the
     PLY / compressed PLY / ZIP64 / chunked-transfer parsers (F4, F8, F12, F13, F23); confine `remote-job.json`
     paths (F24) and benchmark datasets (F19).
   - Then a fresh Claude Security scan on the result.
2. **Fuzzing of the parsers** (PLY, SPZ, ZIP, dataset.json, COLMAP binaries) on the Mac and in splatprep. These run
   on files people download from the internet.
3. **A Mac test target** (`swift test`): parsers, dataset building, settings catalogue, the upload batching and
   resume logic, project checkpoints. Run it in CI next to the Linux suite.
4. **Copy diagnostics** in every app: one button that zips the project log, app / OS / GPU / driver versions,
   worker list and the last error, with paths and the user name removed, for attaching to an issue. No telemetry,
   which keeps the "nothing leaves your network" promise.
5. **Commercial-safe defaults:** benchmark ALIKED + LightGlue against SuperPoint on the sample sets. If it is
   within reach, make it the default and leave SuperPoint as a download with its licence shown. Use the heuristic
   sky mask (or a permissively licensed segmenter) by default. Say plainly on the website which parts are
   non-commercial.

Done when: no open high or medium findings; parsers fuzzed for a night without crashes; Mac tests in CI;
diagnostics in all three desktop apps; defaults are licensed for commercial use.

## Phase 2: one product on every platform (3–4 weeks)

The Linux and Windows apps share one GTK interface. Bring it level with the Mac so the docs describe one product.

1. Projects list with stage checkpoints and Redo / Resume; drone mode and the photo switches (Leave out take-off
   shots, sky, GPS); Show Live View; block Retry / Leave Out / Pause; the new preparation screen; Upload size.
2. VHS clip list with pictures (now numbers on Linux); DJI `.SRT` locations for drone video.
3. Fix the remaining known issues: cancelled multi-device runs leaving photos on workers; the NVIDIA Flatpak
   driver extension after a driver update (offer to install it, not just show the command).
4. Real-hardware passes, written up in `docs/dev/` like the AMD test: Windows PC with an NVIDIA GPU (borrowed or a
   cloud GPU instance), iPhone 1.2 on a phone, a three-device block run on a real network.

Done when: the user guide needs no "Mac only" notes except Metal and Object Capture; each platform has passed a
written real-hardware test.

## Phase 3: distribution (1–2 weeks, partly waiting on accounts)

1. **Mac:** Developer ID signing and notarization. The release workflow already does it when the secrets are set,
   so this needs an Apple Developer Program membership and the secrets.
2. **Windows:** sign the MSI and the executables (Azure Trusted Signing or an OV certificate) so SmartScreen
   doesn't block it. Add an update check like the Mac's `Updater`.
3. **Linux:** pick a real Flatpak id (reverse DNS of a domain you control, or `io.github.Benevision.SplatWork`),
   ship the old id with `end-of-life-rebase` so existing installs move over, then submit to Flathub. Keep the
   Bonjour names and the Keychain service as they are.
4. **iPhone:** TestFlight first (needs the same Apple membership), App Store later.
5. **Website:** a download page per platform with hardware requirements (GPU, VRAM, OS), checksums, and "which
   app do I need" in a few lines.

Done when: each platform installs without a security warning, and every app tells the user about updates.

## Phase 4: first-run experience and docs (2 weeks, can overlap phase 3)

1. **First run:** a hardware check (GPU, memory, disk), the model downloads with their sizes and licences, and a
   **Try a sample** button that makes a splat from a small bundled or downloaded set, so a new user sees success
   before using their own photos.
2. **Capture guide:** how to shoot for a good splat (overlap, light, motion blur, drone patterns), with pictures.
   Link it from the error messages that come from bad captures (few photos placed, GPS mismatch).
3. **Error message pass:** each failure says what happened and what to try (the drone errors are the model).
4. **Troubleshooting page:** workers not found (firewall, VPN, Tailscale), NVIDIA driver extension, out of
   memory, slow uploads.
5. **Accessibility:** VoiceOver labels and keyboard paths through New Project, training and export on the Mac;
   the same check for GTK with Orca.
6. Localization: make strings localizable now (`String(localized:)`, gettext), translate later if users ask.

## Phase 5: public beta, then 2.0 (4–6 weeks of real use)

1. Release **1.3.0** after a fresh dry run, marked as a public beta on the website.
2. Invite 10–20 testers with different hardware (Apple silicon generations, RTX 30 / 40 / 50, an AMD card, Windows
   and Linux), each with a short test script and the diagnostics button.
3. Weekly triage of issues; patch releases as needed. Watch for the same failure from several people.
4. **2.0 when:** two weeks with no new crash or data-loss reports; every platform passed its hardware test; the
   docs match the apps.

## After 2.0

From the original roadmap, in the order the architecture was prepared for them:

- Steam Frame capture (stereo + IR cameras together, for room meshing).
- Vulkan trainer: densification on the GPU, streaming datasets bigger than VRAM; a Windows build.
- Multi-source datasets (drone + phone, 360° + Steam Frame).
- Temporal / 4D splatting.

## Decisions for you

These change the plan or cost money, so they're yours:

1. **Apple Developer Program** ($99 / year): notarized Mac app and TestFlight for Capture. Strongly recommended;
   without it every Mac user sees a Gatekeeper warning.
2. **Windows code signing:** Azure Trusted Signing (about $10 / month) or a certificate. Without it SmartScreen
   warns on every download.
3. **New Flatpak id:** needed for Flathub. Your rule so far is that the id never changes; a migration with
   `end-of-life-rebase` keeps existing installs working.
4. **Commercial use:** should SplatWork be safe for businesses by default (permissive models), or stay
   personal / research use with the best models on by default?
5. **Who it's for:** hobbyists who download apps, or developers who build from source. This decides how much of
   phases 3 and 4 is needed before the beta.
