# Current status (read after CLAUDE.md; details in docs/lab-notes.md)

Last updated 2026-10-28. Short hand-off so a new session can pick up without the old conversation.

## Where things stand
- **Bear 2 (IMG_6705, seated cowboy-hat bear, 16 cm, concrete)** ran end to end with ONE command on 2026-10-27:
  `scripts/runs/run_full_6705.sh` (log `logs/IMG_6705_full.log`, load `logs/IMG_6705_load.csv`), 1 h 52 min, no hand values.
  Final `exports/IMG_6705_final/splat_shadowfree.ply`; renders `exports/IMG_6705_final_compare/`.
- Worked: auto object finding + scale (`scripts/find_object.py`, K = 0.767 vs bear 1), sun elevation from video metadata
  (`scripts/sun_from_metadata.py`, 44.47 deg; no compass heading in the file, so azimuth is fitted from the shadow with
  `fit_sun_edges.py --fix-elevation`), ground shadow removal (texture fill U + splat clone stamp Y).
- **Failed: object (bear) fill**: bear turns orange / glassy. Cause 1: `--adj-radius` scaled to 1.72 units (18 cm) is
  bigger than the bear, so the adjacent rule compares each spot with the whole bear; paint is read as shade.
  Cause 2 (root): standard 3DGS blobs have no reliable surface direction (axis vs PCA normals 42 deg apart, ~1 in 4
  confident), so the fill cannot use geometry (sun-blocked + facing) and falls back on brightness.
- Other open issues: predicted ground shadow ~3x too big in stage 4a (IoU 0.34); receiver offset at the scan edge in both
  sun fits; grey smudge at the bear's feet (127 unfilled clone cells); own sun fit +7 deg without metadata.

## Decided next step (user, 2026-10-28): surface test
- Train a **2DGS** (flat surfel) splat of bear 2 on the EXISTING poses (`data/IMG_6705`, no COLMAP rerun); gsplat 1.4.0 in
  env `gsplat` has `rasterization_2dgs`. About 45-60 min.
- User does NOT want lower visual quality: plan is two splats, normal splatfacto splat for looks, 2DGS only as a geometry
  reference (each normal blob borrows the nearest surfel's normal). Measure before deciding: held-out PSNR/SSIM/LPIPS of
  both + crops of the hat; share of blobs with a confident normal.
- After that: geometry-driven fill = brightness x E_lit / E, E = sky + sun * max(0, n.s) * visible, with the sun/sky
  ratio measured on the concrete next to the object (not fitted on the painted bear).

## Pending decisions / reminders
- Commit the new uncommitted scripts? (`sun_from_metadata.py`, `find_object.py`, `sample_load.py`,
  `runs/run_full_6705.sh`, `fit_sun_edges.py --fix-elevation`; plus older uncommitted bear 1 scripts). Ask first.
- Remind the user: GLOMAP (`colmap4` env, `global_mapper`) vs incremental mapping timing test. Not queued.
- User may film new videos (AE/AF lock, slower, 2-3 loops incl. one high, own shadow off the scene).

## House rules (from the user)
- One command at a time; nothing new without the user's yes; ask before commits; log every step in lab-notes.
- Report each scene on its own, comparisons only at the end. Ground shadows must be removed too.
- The adjacent brightness-and-colour rule is the user's own method: credit it.
- Visible terminal: tmux `splat`, window `bear2`; read-only ttyd (port 7681) + cloudflared link in `~/cloudflared_ttyd.log`.
