# Scripts, by pipeline stage

Stage numbers follow the paper (run order).
`runs/` holds the exact run scripts behind each result in `docs/lab-notes.md`;
`runs/run_full_6705.sh` runs the whole pipeline on one video.

## Stage 1: video to splat
| Script | What it does |
|---|---|
| `setup_env.sh` | One-time environment setup (conda, nerfstudio, COLMAP, patches). |
| `make_splat.sh`, `make_splat_hq.sh` | Video to trained, exported splat (8 GB defaults / high-quality full-resolution). |
| `select_sharp_frames.py` | Sharpest frame from each window of a video. |
| `fix_colmap_model.py` | Use the largest COLMAP sub-model, keep COLMAP coordinates. |
| `hloc_sfm.py` | Alternative structure-from-motion with SuperPoint + LightGlue. |
| `find_object.py` | Find the main object in the splat and the scene scale from its real height. |
| `train_2dgs_ref.py` | Flat-surfel (2DGS) copy of the scene, used only as a surface-direction reference. |
| `borrow_normals.py`, `normals_picture.py` | Give each blob the nearest surfel's direction; picture of the directions. |
| `train_ppisp_ft.py` | Fine-tune a splat so its colours become camera-independent linear light. |

## Shadow votes (2D shadow masks onto the splat)
| Script | What it does |
|---|---|
| `sddnet_masks.py`, `sddnet_masks_ckpt.py` | Shadow masks for every frame with SDDNet (original or fine-tuned weights). |
| `make_chair_targets.py`, `finetune_sddnet.py` | Build training targets from the 3D labels and fine-tune the detector. |

## Stage 2: surfaces and grain
| Script | What it does |
|---|---|
| `cluster_surfaces.py` | Group blobs into continuous surfaces and measure each surface's grain direction. |
| `check_grain.py` | Check the grain against the texture direction in the photos. |

## Stage 3: sun direction
| Script | What it does |
|---|---|
| `fit_sun.py` | Sun direction from cast-shadow area overlap. |
| `fit_sun_edges.py` | Sun direction from the shadow outline (boundary chamfer score). |
| `fit_sun_self.py` | Adds the object's shadow on itself. |
| `sun_from_metadata.py` | True sun position from the video's GPS and capture time, for checking. |
| `sun_refit.py` | Refit the sun through the 2DGS surfels, with only the object casting. |

## Stage 4: visibility (which blobs are in shadow)
| Script | What it does |
|---|---|
| `shadow_visibility.py` | Flag blobs the sun cannot reach (object and ground). |
| `sun_shadow_map.py` | Fine shadow map rendered from the sun through the surfels. |
| `detect_object_shadow.py` | Shadow on the object from its own brightness and the adjacent brightness-and-colour rule. |
| `shadow_overlay.py` | Draw the shadow labels over the real photos. |

## Stage 5: fill
| Script | What it does |
|---|---|
| `fill_shadows.py` | Same-surface, grain-guided repainting of shadowed object blobs. |
| `fill_geometry.py`, `fill_labels.py` | Later object fills driven by surface direction and sun-map labels. |
| `relight_sh.py` | Experimental: split blobs into paint x light and relight. |
| `fix_splat_shadows.py` | Early ground fix: brighten blobs in cast shadow. |
| `clone_ground_splats.py` | Ground clone stamp: replace shadowed ground blobs with copies of sunlit ones. |

## Evaluation, rendering and export
| Script | What it does |
|---|---|
| `render_splat_compare.py`, `render_orbit.py`, `crop_compare.py` | Side-by-side renders against the photos and orbit views. |
| `gsir_export_albedo.py` | Export a GS-IR result as a normal splat (baseline comparison). |
| `ply_to_splat.py`, `web_orient.py`, `web_st_args.py`, `render_upright_check.py` | Web-viewer export helpers. |
| `setup_relightable_splat.sh`, `train_relightable_splat.sh` | GaussianShader / Relightable 3D Gaussian baselines. |
| `watch_nerfstudio.sh`, `watch_gaussianshader.sh`, `sample_load.py` | Progress and machine-load monitors for long runs. |
