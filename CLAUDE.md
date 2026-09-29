# Quick reference for Claude (read this first)

Short summary of how this project is set up, what we changed, and what has broken before.
Full details, dates and numbers are in `docs/lab-notes.md`. Update both when something changes.

## Machine
- Home laptop `Murph`: WSL2 Ubuntu 26.04, RTX 4070 Laptop **8 GB VRAM**, **15.5 GB RAM visible to WSL**
  (Windows has 32 GB), 32 cores. Windows user folder is `/mnt/c/Users/roach/` (the videos are in `Downloads/`).
- Conda: `~/miniforge3`. Main env `gsplat` (nerfstudio, COLMAP 3.10, PyTorch 2.11+cu128).
  Separate env `colmap4` (COLMAP 4.1.1 with `global_mapper`, not used in the pipeline yet).
- The user usually drives this remotely from school via **Claude Remote Control**
  (`claude remote-control`, tmux session `claude`). They cannot see the terminal.

## How the user wants to work
- Send progress updates about every 5 min on long runs and alert right away on errors or crashes.
  Use `watch_600f.sh PID LOG [DONE_PATTERN]` through the Monitor tool, filtering out harmless viewer
  AssertionErrors, and re-arm it every 30 min.
- Log every change, run and problem in `docs/lab-notes.md`.
- Ask before big or unexpected actions: they often interrupt tool calls they didn't expect.
- Screenshots / SendUserFile do not work in this session. Show terminal contents as text instead.

## Standard pipeline (one video -> one scene)
`./scripts/make_splat_hq.sh <video> <scene> [num_frames=650] [iters=30000]`
1. `scripts/select_sharp_frames.py`: sharpest frame per window.
2. `ns-process-data images` (COLMAP).
3. `scripts/fix_colmap_model.py`: use the **largest** COLMAP sub-model, refine intrinsics, keep COLMAP coordinates.
4. `ns-train splatfacto-big` at full res, `cache-images cpu`, pose optimisation `SO3xR3`,
   `stop-split-at ${STOP_SPLIT:-11000}`, `--orientation-method none --center-method none --auto-scale-poses False`.
   Falls back to `splatfacto` if it fails.
5. `ns-export gaussian-splat` -> `exports/<scene>/splat.ply`.
Steps 1-3 are skipped if `data/<scene>/transforms_nsprocess.json` exists.

## Known problems -> rules (details in lab-notes)
1. COLMAP can split the scene and nerfstudio reads only `sparse/0`. **Always check the frame count in `transforms.json`.**
2. Nerfstudio re-orients the scene by default, so the splat doesn't match the SfM points. Use the flags in step 4.
   Side effect: in the viewer, use "Reset Up Direction".
3. Viewer `AssertionError` in websockets `_drain_helper` is harmless (browser connect or disconnect).
4. **Never use `--eval-mode all`**: it caches the frames twice and nearly ran out of RAM. About 650 full-res frames is
   about 10-11.6 GB RAM, near the limit.
5. **splatfacto-big fills the 8 GB of VRAM before step 15k** (7.2 GB at 8.2k). Then WSL spills into system RAM and slows
   to a crawl, or crashes with "CUDA driver error: device not ready" (IMG_6557 at step 7390, even with stop-split-at 10000).
   On this laptop use `stop-split-at` <= 7000 for big, or plain splatfacto. Monitor re-arms can lag, so check the log directly after each expiry.
6. Resuming needs `export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1` (PyTorch 2.11 torch.load default).
7. **Resuming mid-densification crashes** (CUDA "index out of bounds"). Resume with `stop-split-at` <= the checkpoint step.
7b. When resuming, `--max-num-iterations` counts **extra** steps from the checkpoint (10k + 30000 = 40k). Pass the remaining steps only.
8. Runner scripts must `exit` on failure. One once carried on to the next video after a failed step.
9. Don't `pkill -f <pattern>` when the pattern appears in your own command line: it kills your own shell. Use PIDs or `[x]yz` patterns.
10. The laptop sleeps if the lid is closed or it's unplugged (keep-awake doesn't prevent that). GPU hits 86 C during COLMAP matching.
11. COLMAP 4 can't open COLMAP 3.10 databases. If you use `colmap4`, run the whole pipeline with it.
12. Remote Control login can expire (`OAuth ... revoked`). Only fixable by running `/login` at the laptop.
13. Checkpoints are saved every 2000 steps and only the latest is kept. Nerfstudio names the folder `splatfacto` even for splatfacto-big.
14. `git push` fails from the laptop (no GitHub credentials). Commits stay local. Git identity: `-c user.name=Claude -c user.email=noreply@anthropic.com`.

## Background services (may or may not be running)
- `~/.local/bin/keepalive.sh` writes to `~/keepalive.log` every 5 min. Hidden PowerShell keep-awake.
- Viewer: `localhost:7007`. Public: `cloudflared tunnel --url http://localhost:7007` (new random link each start).
- Terminal dashboard: tmux `dash`, served read-only by `ttyd` on port 7681 (user `murph`, password in `~/.ttyd_pass`) plus a second cloudflared tunnel.
- Stop public access: `pkill -f "[c]loudflared tunnel"; pkill -f "[t]tyd -p"`.

## Scenes so far
| Scene | Source | Best result |
|---|---|---|
| `my_scene` | IMG_6449 (313 frames) | `outputs/my_scene/splatfacto/2026-09-16_204930` |
| `my_scene_600f(_hq)` | IMG_6449 (659 frames) | `exports/my_scene_600f_hq/splat.ply` (30k, full res, COLMAP-aligned) |
| `IMG_6556` | IMG_6556.MOV | Done 2026-09-28: splatfacto-big, 40k steps (resumed at 10k, stop-split-at 10000), 2.89M Gaussians, `exports/IMG_6556/splat.ply` (717 MB), config `outputs/IMG_6556/splatfacto/2026-09-28_151253/` |
| `IMG_6556` relightable | GaussianShader on `data_gaussianshader/IMG_6556` (undistorted) | Done 2026-09-29 01:18 (`run_gs_6556.sh`: 30k iters, `-r 2`, `--data_device cpu`, `--densify_grad_threshold 0.0004 --densify_until_iter 7000`, env `gaussian_shader`). Train PSNR 25.06. `outputs_relightable/IMG_6556/point_cloud/iteration_{5..30}000/point_cloud.ply` (about 106 MB). GPU about 3.1 GB after densification. Watch runs with `watch_gs.sh PID LOG 30000`. |
| `IMG_6557` | IMG_6557.MOV | Done 2026-09-29 10:58. splatfacto-big crashed at step 7390 ("CUDA driver error: device not ready"), and the automatic fallback to splatfacto finished 30k: `exports/IMG_6557/splat.ply` (1.45M Gaussians, 359 MB), config `outputs/IMG_6557/splatfacto/2026-09-29_100714/`. |
