# Session report: Gaussian Splat training on a home laptop, driven remotely (Sept 23-29, 2026)

This is a self-contained summary of a multi-day working session between the user and Claude Code, which ran on
the user's home laptop and was controlled remotely (mostly from school) through Claude Remote Control. It is meant as
source material for writing up observations. Companion files in the repo: `CLAUDE.md` (quick reference) and
`docs/lab-notes.md` (detailed log with all numbers).

---

## 1. Setup and context

**Hardware:** Windows laptop "Murph", WSL2 Ubuntu 26.04, NVIDIA RTX 4070 Laptop GPU (**8 GB VRAM**), 32 CPU threads,
32 GB Windows RAM, of which **15.5 GB is visible to WSL**, 900 GB free disk.

**Software:** nerfstudio (splatfacto / splatfacto-big) + COLMAP 3.10 in conda env `gsplat` (PyTorch 2.11 + CUDA 12.8).
Added during the session: `gaussian_shader` env (GaussianShader, PyTorch 2.4 + CUDA 12.4), `colmap4` env
(COLMAP 4.1.1 with the built-in GLOMAP global mapper), cloudflared, ttyd and the GitHub CLI.

**How the user worked:** they started `claude remote-control` in a tmux session on the laptop, then gave instructions
from the Claude app at school. They couldn't see the laptop's terminal, so everything had to be reported in the chat.

**Source videos:** iPhone clips, 1920x1080 @ 60 fps.
- `IMG_6449.MOV`: 99 s, 5,929 frames (scene: `my_scene`)
- `IMG_6556.MOV`: 104 s, 6,221 frames (outdoor: blue and red Adirondack chairs on grass, fence, sun and shade)
- `IMG_6557.MOV`: 90 s, 5,376 frames (separate scene)

---

## 2. Timeline

### Day 1 (Sept 23): remote access, viewing, first retrains
1. **Remote access check:** Claude can run commands on the laptop, but can't type into the user's own terminal window.
2. **Viewing from school:** set up a Cloudflare quick tunnel (random public `*.trycloudflare.com` link) to the
   nerfstudio viewer on `localhost:7007`. Checked that both the page and the websocket connection work from outside.
   Later the user asked to close it and use localhost only (privacy preference).
3. **Keeping the laptop awake:** the user asked for "a command every 5 minutes". Claude explained that terminal
   inactivity isn't what drops the connection; sleep is. It set up a 5-minute heartbeat log plus a Windows keep-awake
   process.
4. **Retrain `my_scene` at 30k steps** (stopped at 64% to make room for the next request).
5. **"More frames, small iterations, don't bottleneck GPU RAM":** 600 frames, 7k steps, frames cached in CPU RAM.
   - **Silent failure found:** COLMAP split the scene into 2 sub-models. nerfstudio used `sparse/0`, which held
     **2 of 659 frames**, printed a warning and trained anyway. Claude caught it by checking `transforms.json`, stopped
     training at 84%, and rebuilt from the 659-frame model.
6. **"Need more quality; SfM and splat don't match":** Claude asked which kind of mismatch; the user said "both".
   - Full resolution (nerfstudio silently caps images at 1600 px wide), 30k steps.
   - Kept COLMAP's coordinate system (nerfstudio normally re-orients, re-centres and re-scales the scene).
   - **Verified alignment numerically:** median distance from splat centres to the nearest SfM point was 0.058 units
     in a scene about 5-7 units across.
   - Side effect the user noticed: the viewer's orbit controls changed. Fix: "Reset Up Direction" in the viewer.
   - Result: 595k Gaussians, 148 MB, 54 min, peak GPU about 2.2 GB.
7. **Progress monitoring:** after a previous silent crash, the user asked for continuous updates. Claude built a
   watcher (status every 15 s to a file, a chat update every 5 min, instant alerts on stage change, errors and crashes)
   and saved "always do this on long runs" to its memory.

### Days 2-6 (Sept 24-28): two new videos, high-quality pipeline
8. **New goal: "very crisp."** New pipeline `scripts/make_splat_hq.sh`:
   - Picks the **sharpest frame** in each window (Laplacian variance) instead of evenly spaced frames.
   - Automatically uses the **largest COLMAP sub-model** (prevents the day-1 silent failure).
   - splatfacto-big at full resolution with camera-pose refinement, falling back to splatfacto if it fails.
9. **The user clarified "two separate scenes", not one merged scene.** Claude ran them one after the other because of
   the RAM limit.
10. **GLOMAP detour:** Claude said GLOMAP could speed up COLMAP. The user pointed out that GLOMAP is now merged into
    COLMAP. Claude **first wrongly said conda-forge had no new COLMAP** (it had read the oldest entries of the list), then
    corrected itself: installed COLMAP 4.1.1 with `global_mapper` in a separate env. It was never used in production;
    COLMAP 4 can't open COLMAP 3.10 databases.
11. **Screenshots:** Claude could take screenshots of the Windows desktop but **couldn't deliver files** in this session.
    It showed terminal contents as text instead, then set up a **read-only, password-protected web terminal** (ttyd)
    through a second tunnel. Later it opened a local Windows Terminal window showing the dashboard.
12. **IMG_6556 training incidents:**
    - **RAM nearly exhausted:** `--eval-mode all` made nerfstudio cache all 650 full-res frames twice (11.8 GB, 0.7 GB
      free, swap full). Claude **stopped it before WSL crashed**, removed the flag and restarted, reusing COLMAP.
    - **GPU nearly full:** splatfacto-big reached 7.2/8 GB by step 8.2k and slowed from about 40 to 560 ms/step. The user
      chose "let it run", then asked to "resume from the checkpoint to free VRAM".
    - **Resume failure 1:** PyTorch 2.11 `weights_only` refuses nerfstudio checkpoints. Fixed with
      `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1`. The runner also carried on to the next video after the failure; Claude
      stopped it and made scripts exit on failure.
    - **Resume failure 2:** resuming in the middle of point-adding crashes ("CUDA index out of bounds"). Fixed by
      stopping point-adding at the checkpoint step. After that: 6.1 GB, 144 ms/step.
    - **Resume gotcha:** `--max-num-iterations 30000` counts extra steps, so it trained to **40k** (131%).
    - **Viewer slowdown:** having the viewer open during training filled the GPU (7.9 GB) and slowed steps about 3x.
    - Result: **2.89M Gaussians, 717 MB**, the most detailed splat of the session.
13. **The user asked for a quick-reference file for Claude:** created `CLAUDE.md` (auto-loaded by Claude Code), plus
    `docs/lab-notes.md` as the full change and problem log. Claude saved "keep lab notes updated" to memory.

### Days 6-7 (Sept 28-29): relightable splat (GaussianShader), IMG_6557
14. **"Run BRDF on the training we just did."** The repo's GaussianShader setup had never actually been run:
    - Its `environment.yml` is a 2022 snapshot (Python 3.7, torch 1.10) that can't be created. Claude built a modern
      env instead.
    - It needed: conda gcc 12 (system gcc 15 is too new for CUDA 12.4), `libusb` for open3d, the conda CUDA header
      path (`targets/x86_64-linux/include`), and **undistorted PINHOLE** images (`colmap image_undistorter`).
    - First full run: GPU memory grew about 0.5 GB every 2 min and was heading to out-of-memory **before the first
      save**. GaussianShader has no resumable checkpoints. The user chose "restart with limits": fewer new points
      (densify threshold x2), point-adding until 7k only, saves every 5k.
    - Result: **3 h 12 min overnight, PSNR 25.06**, GPU about 3.1 GB, no errors.
15. **Relighting renders:** GaussianShader has no relight flag, so Claude swapped the learned environment map file for
    Poly Haven CC0 HDRs (studio, sunset).
    - **The user found the relit results "too bright with cloudy artifacts."** Diagnosis:
      - Raw HDRs are 3.8-5.4x brighter than the learned light, so Claude made exposure-matched versions.
      - **Key finding:** GaussianShader only relights the **reflection (specular)** part. The diffuse part, with the
        sun, shade and shadows, is baked in and identical in every relight (0.441 in both). On grass, noisy surface
        normals scatter the reflected light, which gives the "cloudy" haze and purple streaks. With matched exposure
        it's clean, but it looks almost identical to the original.
      - Conclusion: GaussianShader is meant for shiny objects and isn't real relighting for an outdoor diffuse scene.
        Alternative suggested: GS-IR.
16. **IMG_6557:** splatfacto-big **crashed at step 7,390** ("CUDA driver error: device not ready", VRAM spill) despite
    the cap and allocator setting. The automatic fallback to splatfacto finished: **1.45M Gaussians, 359 MB**.
    **Monitoring gap:** the watcher expired and the expiry notice arrived late, so the crash was reported 45 min late.
17. **GPU performance estimates** for 4080, 4090, 5090, A100, H100 (see section 4).
18. **GitHub:** the repo is private, and **7 commits exist only on the laptop** (it has no GitHub credentials). A
    device-code login was started, then cancelled by the user.
19. **School machine (RTX PRO 6000 Blackwell, 96 GB):** the source videos were shared through a temporary tunnel link
    and downloaded, and the link was then closed. A second Claude session at school reviewed the repo; see section 6.

---

## 3. Results

| Scene | Method / settings | Gaussians | File | Time | Notes |
|---|---|---|---|---|---|
| my_scene (313 fr) | splatfacto 30k | - | `outputs/my_scene/...204930` | - | From before this session |
| my_scene_600f (659 fr) | splatfacto 7k, half res | - | 92 MB | 3.5 min | Quick test |
| my_scene_600f_hq | splatfacto 30k, full res, COLMAP-aligned | 595k | 148 MB | 54 min | Aligned to SfM (median 0.058) |
| **IMG_6556** | splatfacto-big, 40k (resumed at 10k), full res | **2.89M** | **717 MB** | about 4.5 h incl. incidents | Most detailed |
| IMG_6557 | splatfacto (big crashed), 30k, full res | 1.45M | 359 MB | 51 min training | Source about 30% softer than 6556 |
| IMG_6556 relightable | GaussianShader 30k, half res, limited densify | - | 106 MB per snapshot | 3 h 12 min | PSNR 25.06 |

Relight videos (`exports/IMG_6556_relightable/`): learned lighting, studio and sunset (raw and exposure-matched), plus
two side-by-side comparisons.

---

## 4. Performance and hardware observations

| Job | RTX 4070 Laptop 8 GB (measured) | RTX 4090 24 GB (est.) | RTX 5090 32 GB (est.) | A100 (est.) |
|---|---|---|---|---|
| splatfacto 30k, 650 frames, 1080p | ~50 min, 2-3.4 GB | ~15-20 min | ~12-15 min | ~15-20 min |
| splatfacto-big uncapped | **Doesn't fit** (crash or crawl at 7-8k) | ~35-45 min | ~25-35 min | ~30-40 min |
| GaussianShader full res | Half res only (3.2 h) | ~1-1.5 h | ~45-60 min | ~1-1.5 h |
| COLMAP (650 frames) | 37-50 min (CPU-bound) | about the same | about the same | about the same |

- The limits that shaped every decision: **8 GB VRAM** and **15.5 GB WSL RAM**.
- 650 full-res frames cached in RAM take about 10-11.6 GB. `--eval-mode all` doubles that.
- The GPU hit 86-87 °C during COLMAP matching (at the throttle point) and 75-82 °C during training.
- An open viewer during training cost about 3x speed and 1.8 GB of VRAM.

---

## 5. Problems found (cause, then fix)

1. **COLMAP split the scene, and nerfstudio silently trained on 2 frames.** Always check the frame count in
   `transforms.json`. Now automatic (`fix_colmap_model.py`).
2. **Splat didn't align with the SfM points.** Keep COLMAP's coordinates. Side effect: the viewer's up-axis.
3. **Harmless viewer `AssertionError`** when a browser tab connects or disconnects. Filtered out in the watcher.
4. **`--eval-mode all` doubled RAM**, near a WSL crash. Removed.
5. **splatfacto-big overflows 8 GB.** On WSL it spills into system RAM (crawl or "device not ready") instead of failing
   cleanly.
6. **Resuming:** `weights_only` error; crash when resuming mid-densification; iteration count adds on top of the checkpoint.
7. **Runner scripts continued after a failed step.** Now they exit.
8. **`pkill -f pattern` killed Claude's own shell** (the pattern was in its own command line).
9. **The laptop slept overnight** (lid or battery policy); keep-awake doesn't prevent that.
10. **Remote Control OAuth token expired** once. It recovered, but otherwise it needs `/login` at the laptop.
11. **COLMAP 4 can't open 3.10 databases.**
12. **GaussianShader:** 2022 env unusable, gcc 15 vs CUDA 12.4, conda CUDA header path, PINHOLE-only cameras, no
    checkpoint flag, VRAM growth before the first save, `set -u` breaks conda activation, `render.py` holds 12.5 GB RAM.
13. **GaussianShader relighting only changes reflections.** Baked diffuse and shadows cause the cloudy artifacts on grass.
14. **Monitor expiry notices can arrive late,** which left an unwatched gap (a crash was reported 45 min late).
15. **File delivery (screenshots, videos) doesn't work** in a Remote Control session without a project thread.
    Workarounds: text output, tunnels, ttyd.
16. **git push fails:** no GitHub credentials on the laptop, and the repo is private.

---

## 6. Second-machine review (school: RTX PRO 6000 Blackwell 96 GB, fresh Ubuntu 26.04)

Setting up from the pushed repo (without the 7 local commits), the school session found:
- **COLMAP 3.13 renamed `--SiftExtraction.use_gpu`**. COLMAP is unpinned, so the pipeline breaks on a fresh install.
  Not seen on the laptop (3.10).
- **gcc 15 vs CUDA 12.4** and **conda CUDA headers not found**: the same issues hit on the laptop, fixed only locally.
- **Branch only reachable through the PR** (main has only a README).
- Settings hardcoded for 8 GB; the 1600 px cap is undocumented; contradictory docs; only ffmpeg is pinned.

Laptop session's view: accurate and fair. The COLMAP rename is the most important finding. It reviewed older code (two
issues already have local fixes). It hasn't hit long-run problems yet: the COLMAP split, resume problems, and
GaussianShader's reflection-only relighting. Suggested priorities: (1) pin versions / lockfile, (2) push the HQ pipeline
to main, (3) make setup_env.sh robust on new Ubuntu, (4) GPU settings profiles (8/24/96 GB), (5) doc fixes.

---

## 7. Observations about working this way (human + remote AI agent)

- **Silent failures were the main risk**, more than crashes: 2-frame training, the 1600 px downscale, and RAM or VRAM
  creeping toward a crash with no error. Checking the outputs (frame counts, memory trends, rendered frames compared
  with the ground truth) caught what the logs didn't.
- **Remote-only visibility changed the requirements:** the user couldn't see the terminal, so proactive 5-minute
  updates, crash alerts, status logs for post-mortems, and later a read-only web terminal became essential.
- **The user often interrupted tool calls** they didn't expect (starting the next video, tests). Asking before large or
  irreversible actions worked better than acting first.
- **Claude made mistakes and corrected them:** it misread the conda-forge version list; its `--eval-mode all` addition
  nearly exhausted RAM; it passed an unsupported GaussianShader flag; it wrongly said the viewer would reopen after
  IMG_6556; it misread the resume iteration count; and its watcher had a gap. Each was then disclosed in the chat and
  recorded in the notes.
- **The user supplied domain knowledge the agent lacked:** that GLOMAP was merged into COLMAP, and that the goal was
  "two different scenes".
- **Hardware limits drove most of the engineering effort** (caps, CPU caching, resume tricks). On a 24-96 GB GPU most of
  that work would have been unnecessary; COLMAP time and pipeline robustness would remain.
- **Documentation as memory:** `CLAUDE.md` + `lab-notes.md` let a fresh session (or the school machine) pick up without
  the chat history. The school review showed that a second machine surfaces version drift that one machine never would.
