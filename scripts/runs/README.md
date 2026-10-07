# One-off run scripts (records, not maintained tools)

These are the exact scripts used for specific runs on the RTX 4070 Laptop (8 GB) in Sept 2026. They're kept so
each result in `docs/lab-notes.md` can be traced to the command that produced it. They hardcode scene names,
video paths (`/mnt/c/Users/roach/Downloads/...`) and `cd ~/Claude_Test`, and several wait on each other by
process name, so treat them as recipes to copy from rather than as general tools.

| Script | Run |
|---|---|
| `run_600f.sh` | my_scene_600f: 659 frames, splatfacto 7k, CPU image cache |
| `run_600f_hq.sh` | my_scene_600f_hq: full res, 30k, COLMAP coordinates |
| `run_two_videos.sh` | IMG_6556 then IMG_6557 through `make_splat_hq.sh` (first attempt) |
| `resume_6556_then_6557.sh` | Resume IMG_6556 from step 10k with `stop-split-at 10000` |
| `export_6556_only.sh` | Export IMG_6556 without starting IMG_6557 |
| `run_6557.sh` | IMG_6557 through `make_splat_hq.sh` after the GaussianShader renders |
| `run_gs_6556.sh` | **Working GaussianShader recipe**: env vars for conda CUDA/gcc, `-r 2 --data_device cpu`, limited densification |
| `run_gs_render_6556.sh`, `run_gs_render_matched_6556.sh` | GaussianShader relight renders by swapping `brdf_mlp.hdr`, and MP4s |

General tools live one level up: `scripts/make_splat.sh`, `scripts/make_splat_hq.sh`, `scripts/select_sharp_frames.py`,
`scripts/fix_colmap_model.py`, `scripts/watch_nerfstudio.sh`, `scripts/watch_gaussianshader.sh`.
