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
`wsl --install -d Ubuntu` and watch it through to either the username prompt
or a reboot request, don't close the window early. It needs an
**Administrator** PowerShell only if the WSL feature itself isn't enabled yet.
On machines where IT already enabled WSL (e.g. the school workstation) it
installed without admin rights.

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

## Export fails after a fully successful training run

**`ns-export gaussian-splat` — `_pickle.UnpicklingError: Weights only load
failed` / `Unsupported global: GLOBAL numpy.core.multiarray.scalar`**: this
hits *after* training completes successfully (all 30,000 iterations, config
saved) — only the export/checkpoint-reload step fails. PyTorch 2.6 changed
`torch.load`'s default `weights_only` argument from `False` to `True`, and
nerfstudio's checkpoint loader (`nerfstudio/utils/eval_utils.py`) hasn't been
updated to pass it explicitly, so loading its own checkpoint (which contains
a numpy scalar) gets blocked as a security precaution. Since PyTorch 2.11 is
what `setup_env.sh` installs, this hits reliably.

Fixed by patching that one `torch.load` call to pass `weights_only=False`
(safe here — we're only ever loading our own freshly-trained checkpoint, not
an untrusted file). `setup_env.sh` now does this automatically via `sed`
after installing nerfstudio. If you hit this on an environment set up before
that fix landed, patch it directly and re-run just the export (no need to
redo training — the checkpoint is already saved):
```bash
sed -i 's/torch\.load(load_path, map_location="cpu")/torch.load(load_path, map_location="cpu", weights_only=False)/' \
  "$CONDA_PREFIX/lib/python3.10/site-packages/nerfstudio/utils/eval_utils.py"
ns-export gaussian-splat --load-config <path/to/config.yml> --output-dir <export-dir>
```

## COLMAP problems

- **COLMAP 3.12+ — `unrecognised option '--SiftExtraction.use_gpu'`**:
  `environment.yml` leaves `colmap` unpinned, conda-forge now ships 3.13, and
  3.13 renamed the option to `--FeatureExtraction.use_gpu` (and
  `--SiftMatching.use_gpu` to `--FeatureMatching.use_gpu`). nerfstudio 1.1.5
  still passes the old names. `setup_env.sh` now detects the COLMAP version and
  patches `nerfstudio/process_data/colmap_utils.py` automatically.
- **COLMAP splits the scene, and nerfstudio trains on only a few frames with no
  error.** `ns-process-data` reads only `colmap/sparse/0`, which can hold only
  2 of 659 frames when COLMAP produces several sub-models. It prints a
  "COLMAP only found poses for 0.30% of the images" warning and trains anyway.
  `make_splat.sh` now refuses to train if fewer than 80% of frames are posed.
  Fix with `python scripts/fix_colmap_model.py data/<scene>` (uses the largest
  sub-model).

## Memory limits (8 GB GPU / WSL RAM)

- **Don't use `--eval-mode all`** with `cache-images cpu`: it caches every
  frame twice and nearly exhausted WSL's 15.5 GB RAM with 650 1080p frames.
- **splatfacto-big at 1080p doesn't fit in 8 GB**: VRAM hit 7.2 GB by step
  about 8k. On WSL it then spills into system RAM (a crawl, or `CUDA driver
  error: device not ready`) instead of failing cleanly. Use plain splatfacto,
  or cap densification (`--pipeline.model.stop-split-at 7000`). A 96 GB
  Blackwell ran it uncapped at 13.6 GB peak.
- **Keep the viewer tab closed during training** on small GPUs: an open
  viewer added about 1.8 GB VRAM and slowed steps about 3x.
- **Resuming** (`--load-dir`): needs `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1`,
  crashes ("index out of bounds") if resumed while still densifying (set
  `stop-split-at` <= the checkpoint step), and `--max-num-iterations` counts
  steps on top of the checkpoint.

More detail and numbers: `docs/lab-notes.md`.

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

**Newer Ubuntu (26.04+, what `wsl --install` gives you now) ships gcc 15**, which
CUDA 12.4's nvcc (apt's `nvidia-cuda-toolkit`) refuses to use, so the apt fix
above fails there. Use a conda compiler instead: either a newer CUDA nvcc
from conda-forge (the school machine used CUDA 12.8 nvcc with g++ 14) or
conda's `gxx_linux-64=12`, and point builds at it with
`CC`/`CXX`/`CUDAHOSTCXX` (e.g. `x86_64-conda-linux-gnu-g++`).

**`cuda_runtime.h: No such file or directory` when building gsplat or other
CUDA extensions with conda's CUDA.** conda-forge puts the CUDA headers and
libraries under `$CONDA_PREFIX/targets/x86_64-linux/`, not `$CONDA_PREFIX/include`.
Point the build at them (WSL's GPU driver library lives in `/usr/lib/wsl/lib`):
```bash
T=$CONDA_PREFIX/targets/x86_64-linux
export CUDA_HOME=$CONDA_PREFIX CPATH=$T/include LIBRARY_PATH=$T/lib:$T/lib/stubs:/usr/lib/wsl/lib:$CONDA_PREFIX/lib
```
To make that permanent for the env: `conda env config vars set CPATH=... LIBRARY_PATH=...`.

**Don't use `set -u` around `conda activate`.** conda's gcc and CUDA activation
scripts reference unset variables (`SYS_SYSROOT`, `NVCC_PREPEND_FLAGS`) and the
script dies at startup. Use `set -eo pipefail`, or turn `-u` on after activating.

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
