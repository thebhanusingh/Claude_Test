# Troubleshooting notes

Real issues hit setting this pipeline up from scratch on Windows + WSL2 (RTX 4070
Laptop GPU), and their fixes. Most fixes are already baked into the scripts —
this doc exists so a repeat setup (this machine or a new one) doesn't have to
rediscover them.

## Windows / WSL setup

**`claude` not recognized after installing Claude Code CLI on Windows.**
The installer put it at `C:\Users\<you>\.local\bin\claude.exe` but didn't add
that folder to PATH in the current session. Either run it by full path
(`& "C:\Users\<you>\.local\bin\claude.exe"`), or add it to PATH permanently
with PowerShell's own env var API — **not** `setx PATH "%PATH%;..."`, which is
cmd.exe syntax and will corrupt PATH if run in PowerShell (it doesn't expand
`%PATH%`, it writes it literally):
```powershell
$userPath = [Environment]::GetEnvironmentVariable("PATH", "User")
[Environment]::SetEnvironmentVariable("PATH", $userPath + ";C:\Users\<you>\.local\bin", "User")
```
Then open a **new** terminal window — env var changes don't apply to the
current one.

**`setup_env.sh` needs a real bash, not PowerShell.** Use WSL2. If `wsl`
opens and immediately closes, or `wsl --list --verbose` says "no installed
distributions," the Ubuntu distro didn't actually finish installing — run
`wsl --install -d Ubuntu` from an **Administrator** PowerShell and watch it
through to either the username prompt or a reboot request, don't close the
window early.

**Multi-line paste into a WSL terminal can get garbled** (stray `^[[200~` /
`^[[201~` escape codes leaking through, commands merging into each other).
If that happens, run commands one at a time instead of pasting a block, and
double check your actual working directory (`pwd`) before running anything
destructive — a garbled `cd ~` can silently leave you somewhere unexpected
(e.g. `/mnt/c/WINDOWS/system32`, where `git clone` will fail with a
permissions error since it's a protected Windows directory).

## GitHub auth

Cloning a private repo over HTTPS needs a Personal Access Token as the
password (GitHub dropped plain password auth for git operations years ago).
**Fine-grained tokens threw `remote: Write access to repository not
granted.` / 403 on a plain clone** — a known recurring issue with
fine-grained tokens. A **classic** token with the `repo` scope worked
reliably instead.

## Conda / environment setup

No `conda`/`mamba` in a fresh WSL install → install
[Miniforge](https://github.com/conda-forge/miniforge) (community-maintained,
defaults to conda-forge, no admin needed):
```bash
curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
bash Miniforge3-$(uname)-$(uname -m).sh   # say yes to license + shell init
source ~/.bashrc
```

**Building nerfstudio's dependencies fails with `Permission denied: 'gcc'` /
`g++` not found.** No C/C++ compiler on a fresh Ubuntu WSL install.
```bash
sudo apt update && sudo apt install -y build-essential
```

## Runtime library issues (all now pinned in `environment.yml`)

- **`ffmpeg` — `Unrecognized option 'vsync'`**: nerfstudio's `ns-process-data`
  hardcodes the `-vsync` flag, which conda-forge's latest ffmpeg (9.x) removed.
  Fixed by pinning `ffmpeg=6.*`.
- **`colmap --version` — `libOpenImageIO.so.3.1: cannot open shared object
  file`**: a version-mismatched conda environment (colmap package installed
  before/without a compatible openimageio). Fixed by force-reinstalling both
  together: `conda install -y -c conda-forge --force-reinstall colmap openimageio`.
- **`open3d` import — `libusb-1.0.so.0: cannot open shared object file`**:
  open3d (a transitive nerfstudio dependency, imported even though we don't
  use its Metashape/RealSense features) needs libusb at runtime; its pip
  wheel doesn't bundle it. Fixed by adding `libusb` to `environment.yml`.

## nvcc / CUDA compile issues

Hit this **twice**, in two different code paths, both ultimately the same
root cause: a pip-bundled `nvcc` (from the `cuda-toolkit` pip package pulled
in as an nerfstudio/gsplat dependency) exists but isn't marked executable,
or isn't reliably where the environment expects it.

1. `ns-train`'s `torch.compile`/inductor path called `nvcc --version` while
   writing a compile-failure bug report, masking the real error behind
   `PermissionError: [Errno 13] Permission denied: 'nvcc'`. Fixed by disabling
   torch.compile for training (`TORCHDYNAMO_DISABLE=1`, baked into
   `make_splat.sh` — splatfacto runs fine in eager mode, just marginally
   slower per-iteration).
2. `gsplat`'s own CUDA backend (`gsplat/cuda/_backend.py`) directly checks
   `nvcc` availability to JIT-compile its rasterization kernels on first use,
   and hit the same permission error. `find "$CONDA_PREFIX" -iname "nvcc*"`
   turned up nothing at all — the pip-bundled nvcc wasn't reliably present as
   a real file in this environment. The reliable fix was to stop relying on
   the pip-bundled one and install a real system CUDA toolkit instead:
   ```bash
   sudo apt install -y nvidia-cuda-toolkit
   ```
   This installs an older CUDA compiler (12.4 vs. PyTorch's 12.8 build) but
   that's fine — gsplat just needs a working `nvcc` that can target the
   GPU's compute capability, not an exact version match.

`setup_env.sh` also proactively chmods any `nvcc` it finds under the conda
env after install, in case the pip-bundled one shows up but isn't
executable — but the apt-installed system one is what actually worked here.

**First-time gsplat CUDA compile is silent for several minutes** (`gsplat:
Setting up CUDA with MAX_JOBS=10 (This may take a few minutes the first
time)` with no further output) — this is normal, not a hang.

## Keeping long training runs alive

313 frames × 30,000 iterations is a genuinely long run. Two things kept
interrupting it:
- **Windows sleep.** Set Settings → System → Power & sleep → Sleep → Never
  (both battery and plugged-in), and set lid-close action to "Do nothing" if
  on a laptop, before starting a long run.
- **The terminal window itself closing** (cause not fully pinned down —
  possibly a Windows Update-triggered restart) even while sleep wasn't the
  culprit. Run inside `tmux` so the process survives even if the window
  disappears:
  ```bash
  tmux new -s training
  # ... run the pipeline inside here ...
  # if the window closes, reopen a terminal and reconnect:
  tmux attach -t training
  ```

## Re-running after a partial failure

`make_splat.sh` skips re-running frame extraction/COLMAP (`ns-process-data`,
~6–7 minutes for this video) if `data/<scene>/colmap/sparse/0/cameras.bin`
already exists — useful when debugging just the training step. **Remember to
`git pull` after any fix lands** before assuming a retry has it; a stale
local script silently redoes the whole thing rather than erroring.
