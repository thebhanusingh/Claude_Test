# Quickstart: video → Gaussian Splat → Blender

The complete, ordered path from "I have a video" to "I'm viewing the splat in
Blender," reflecting everything learned getting this working the first time
(see `docs/troubleshooting.md` for the why behind each fix, if curious).
Machine used to validate this: Windows + WSL2, RTX 4070 Laptop GPU (8GB).

## 0. Before you record the video

These matter more than anything downstream — a bad capture can't be fixed in
software:

- **Lock exposure and white balance** if your camera app allows it (many
  phone camera apps let you tap-and-hold to lock AE/AWB). Auto-exposure
  drifting as you move around the subject actively hurts both COLMAP's
  feature matching and splat quality (shows up as floaters/color
  inconsistency).
- **Avoid motion blur** — move slowly and steadily, don't whip the camera.
- **Reflective/glossy subjects (car paint, glass, chrome) are a genuinely
  hard case for Gaussian Splatting** — it assumes smoothly-varying
  view-dependent color, which mirror-like surfaces violate. Denser frame
  coverage helps more than it does for matte subjects; expect some
  shimmer/floaters on reflective panels regardless.
- **Orbit with good overlap** — each frame should overlap significantly with
  the last (don't pan quickly past sections). Full 360° coverage if the
  subject allows it.
- **20-60 seconds of footage** is plenty of raw material; the pipeline
  samples ~300+ frames out of it, not every frame.

## 1. One-time machine setup

Skip to step 2 if this machine already has the environment from a previous
run.

```bash
git clone https://github.com/thebhanusingh/Claude_Test
cd Claude_Test
```

**On Windows**, this needs WSL2 (the `.sh` scripts need real bash):
- `wsl --install -d Ubuntu`, let it run to completion, set a Linux
  username/password when prompted. Use an **Administrator** PowerShell if the
  WSL feature isn't enabled yet; on machines where it already is (common on
  managed lab PCs) it installs without admin.
- Current installs give **Ubuntu 26.04 with gcc 15**, which older CUDA
  compilers reject. See the gcc 15 and `cuda_runtime.h` entries in
  `docs/troubleshooting.md` before building anything with CUDA.
- Clone the repo *inside* WSL's own filesystem (`~/`), not via `/mnt/c/...`
  into a Windows-side folder (avoids OneDrive sync interference and is much
  faster for COLMAP's many small files).
- Install Claude Code CLI, if you want it, with `irm https://claude.ai/install.ps1 | iex`
  in PowerShell — that's for working in a terminal on Windows directly; not
  needed for WSL/bash work.

Then, inside WSL/Linux:
```bash
sudo apt update && sudo apt install -y build-essential   # C/C++ compiler, needed to build some deps
./scripts/setup_env.sh
```
If `conda`/`mamba` isn't installed yet, `setup_env.sh` will fail immediately —
install [Miniforge](https://github.com/conda-forge/miniforge) first:
```bash
curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
bash Miniforge3-$(uname)-$(uname -m).sh   # accept license, say yes to shell init
source ~/.bashrc
```

**If `nvcc --version` isn't found after `setup_env.sh`** (the pip-bundled one
is unreliable), install the real one:
```bash
sudo apt install -y nvidia-cuda-toolkit
```

**Before a long training run**: disable Windows sleep entirely (Settings →
System → Power & sleep → Sleep → Never, both battery and plugged in; lid
action → "Do nothing" if a laptop) — a sleeping machine kills the run.

## 2. Every run: generate the splat

```bash
conda activate gsplat
cd ~/Claude_Test
```

Run inside `tmux` so the process survives if the terminal window closes for
any reason:
```bash
tmux new -s training
```
(if reconnecting to an already-running session instead: `tmux attach -t training`)

Then, inside the tmux session:
```bash
MAX_JOBS=2 ./scripts/make_splat.sh "/path/to/video.mp4" my_scene
```
- `MAX_JOBS=2` limits gsplat's parallel CUDA compile jobs (first-run only,
  can take several minutes with zero output — that's normal, not a hang).
  Lower this further, or drop it, if you have less system RAM; the default
  of 10 parallel jobs caused memory-pressure crashes on a 32GB machine with
  ~15GB available to WSL.
- Add `NUM_FRAMES=600` (env var) before denser subjects/reflective surfaces;
  default is 300.
- Third positional arg overrides iteration count (default 30000):
  `./scripts/make_splat.sh video.mp4 my_scene 20000`

Detach with `Ctrl+b` then `d` if you want to close the window without
stopping training; reattach later with `tmux attach -t training`.

**Watch it live**: open a Windows browser to `http://localhost:7007` (WSL2
forwards this automatically) to see the nerfstudio viewer while it trains.

**Re-running after a partial failure**: the script skips re-extracting
frames/COLMAP if `data/<scene>/colmap/sparse/0/cameras.bin` already exists —
useful when only the training step needs retrying. **Always `git pull` after
any repo fix lands** before assuming a retry has it.

## Optional: drive the home PC remotely (e.g. from a university PC)

Uses Claude Code Remote Control — nothing to install on the remote/uni PC,
and it works through restrictive firewalls since the home PC only makes
outbound connections.

On the home PC, **inside WSL** (that's where the `gsplat` env and scripts
live; a Windows-side Claude Code install can't run them directly):
```bash
curl -fsSL https://claude.ai/install.sh | bash   # one-time
tmux new -s claude
cd ~/Claude_Test
claude remote-control
# Ctrl+b then d to detach; leave it running
```
Keep the PC plugged in, Sleep → Never, lid open. A Windows Update reboot kills
it — restart the steps above afterward (`tmux attach -t claude` if the tmux
session survived).

From anywhere else: open claude.ai/code (or the Claude mobile app) — the home
PC's session appears there.

Limits:
- `localhost:7007` (the nerfstudio viewer) is only reachable *on* the home
  PC. To see it remotely, use a remote desktop tool (e.g. Chrome Remote
  Desktop, browser-based).
- New videos have to get onto the home PC — easiest is a synced folder
  (OneDrive/Google Drive); e.g. `C:\Users\<you>\OneDrive\Videos` is
  `/mnt/c/Users/<you>/OneDrive/Videos/` inside WSL. Heavy I/O straight off a
  synced folder is slow, so copy the video into WSL (`~/`) before running.

## 3. View / use the result

**For just looking at it: prefer the nerfstudio viewer (localhost:7007) over
a web `.ply` viewer like SuperSplat** — confirmed better/more familiar
navigation controls than SuperSplat's orbit/pan scheme. It closes
automatically when training finishes (`--viewer.quit-on-train-completion
True`); reopen it against the trained config:
```bash
ns-viewer --load-config outputs/<scene_name>/splatfacto/<timestamp>/config.yml
```
Then open `http://localhost:7007` in a Windows browser (WSL2 forwards this
automatically). Find `<timestamp>` with:
```bash
ls outputs/<scene_name>/splatfacto/
```

**For editing/compositing/importing into a larger scene**, use the exported
`.ply` at `~/Claude_Test/exports/<scene_name>/splat.ply` (Windows path:
`\\wsl.localhost\Ubuntu\home\<you>\Claude_Test\exports\<scene_name>\splat.ply`).
Load it in Blender rather than a web viewer if you already know Blender's
navigation:

1. Get [KIRI Engine's "3DGS Render" addon](https://github.com/Kiri-Innovation/3dgs-render-blender-addon)
   (free, open source) — download the release zip.
2. Blender: `Edit → Preferences → Add-ons → Install...`, pick the zip, enable it.
3. New "3DGS Render" tab in the `N` sidebar → **Import PLY** → select `splat.ply`.
   Use "Import as Points" first for fast positioning, then switch to full
   splat rendering.

See `docs/lab-setup.md` for the equivalent Unreal Engine path, and
`docs/troubleshooting.md` for anything that goes wrong along the way that
isn't covered above.
