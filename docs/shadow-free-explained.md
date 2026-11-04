# Shadow-free Gaussian splats: the pipeline explained

*Plain-language companion to `docs/lab-notes.md`. All 20 test images with captions: [figures README](figures/2026-10-12_chair_shadow_tests/README.md). Scene: IMG_6556, an Adirondack chair on grass in sunlight, filmed
on a phone on 2026-09-28 at 12:58 local time. Status as of 2026-10-13. The numbers and settings for every run are in the
lab notes.*

## Background: what a Gaussian splat is

A **Gaussian splat** is a way to store a 3D scene that was captured on video. Instead of a mesh of triangles, the
scene is millions of small, soft, coloured blobs ("Gaussians"). Each blob has a position, a size and stretch in three
directions, an orientation, a colour and a transparency. Seen from any viewpoint, the blobs blend into a photo-like
image. This scene has about 2.9 million of them; the chair alone is about 24,000.

**The problem.** The blobs record the colours the camera saw, *including the shadows*. A shadow under the chair is
literally painted onto the grass blobs, and a shaded slat is painted dark. If the scene is reused in Blender or Unreal
with different lighting, those baked-in shadows look wrong.

**The idea of this project (the brief).** Instead of fully modelling materials and light physics, which is hard and
slow, treat shadow removal as *repainting*. Find which blobs are in shadow, then recolour each one from lit blobs on the
**same surface**, following the direction of the surface's texture (its "grain"). The brief has five stages.

## Stage 1: Capture and 3D reconstruction (done, before this brief)

**What:** turn a phone video into a splat.
**How:** pick about 650 sharp frames. **Structure from Motion** (COLMAP) works out where the camera was for each frame
by matching features between photos. Then train the splat (nerfstudio, "splatfacto") so that its renders match the
photos.
**Result:** `exports/IMG_6556/splat.ply`, 2.9 million Gaussians. It's kept in COLMAP's coordinates so later stages
can line up 3D points with the original photos.

## Stage 2: Estimating the sun direction (done; result 55.1 deg vs true 55.5 deg)

**What:** find where the sun was, as an elevation angle above the ground and a compass direction.
**Why:** stage 4 needs it to decide which blobs the sun could reach.
**How, in plain words:** for a guessed sun direction, cast the chair's shadow onto the ground in 3D, like holding a
torch over a model. Compare that predicted shadow with where the photos actually show shadow. Try many directions;
the best match wins. The "photos show shadow" part comes from **SDDNet**, an existing neural network that marks
shadow pixels in each photo. Each ground blob is looked up in every photo that sees it and gets a shadow/lit vote.
As the brief says, nothing reads the sun from the blobs' colour coefficients.

**Ground truth for checking:** the video's metadata gives time and GPS, and a sun calculator gives 55.51 deg elevation
(about +/-0.1 deg). The compass direction can't be checked, because the reconstructed scene has no north.

**How the estimate improved (each step logged in the lab notes):**

| Step | What changed | Elevation | Error |
|---|---|---|---|
| 2.1 | First fit: every object in the area casts shadows | 57.0 | +1.5 |
| 2.2 | Only the chair casts (grass blades and noise removed) | 55.0 | -0.5, but jumpy across settings |
| 2.3 | Also score the chair's shadow on itself | failed alone; parked (see below) | |
| 2.4 | Compare shadow *outlines* instead of areas | 56.3 | +0.8, steady |
| 2.5 | Test: is the grass surface at the right height? | showed the shadow lies on the grass *tips* | |
| 2.6 | Fit the sun and the grass-tip height together | **55.08** | **-0.43** |

Reported result: **55 deg +/- 1 deg** (one scene so far). Why 2.3 failed: the backrest slats are only 2-3 cm thick, and
the visibility test couldn't tell a slat's front face from its back face, so the back faces inherited "lit" votes.
**Not done yet:** the brief also suggests using shading gradients (how brightness changes with surface angle) as a
second cue.

![Figure 2.1: first fit, everything casts. Observed, predicted, agreement (57.0 deg)](figures/2026-10-12_chair_shadow_tests/s2_01_sun_fit_all_casters.jpg)
*Figure 2.1: first fit, everything casts. Observed, predicted, agreement (57.0 deg)*

![Figure 2.2: only the chair casts (55.0 deg)](figures/2026-10-12_chair_shadow_tests/s2_02_sun_fit_chair_only.jpg)
*Figure 2.2: only the chair casts (55.0 deg)*

![Figure 2.4: outline scoring (56.3 deg)](figures/2026-10-12_chair_shadow_tests/s2_04_sun_boundary_fit.jpg)
*Figure 2.4: outline scoring (56.3 deg)*

![Figure 2.6: final fit with grass-tip height (55.08 deg vs true 55.51)](figures/2026-10-12_chair_shadow_tests/s2_05_sun_boundary_fitted_grass_offset.jpg)
*Figure 2.6: final fit with grass-tip height (55.08 deg vs true 55.51)*

## Stage 3: Grouping blobs into surfaces, with a grain direction (done)

**What:** split the chair's blobs into continuous surfaces (seat, each backrest slat group, arms, legs), and find
the texture direction on each.
**Why:** stage 5 must only borrow colour from the *same* surface, and along its grain, so wood streaks continue
instead of smearing.
**How:** each blob is a flattened ellipsoid; its shortest axis acts as a surface normal (the direction the surface
faces). Neighbouring blobs join a group while their normals stay within 30 deg of the group's average. That's region
growing, like flood-fill on a map. Colour isn't used, so a shadow edge can't split a surface. The grain comes from
the direction each blob is *stretched* in: training stretches blobs along visible texture.
**Problems solved:** real splat normals are noisy (24-45 deg off), and the first version chained the whole chair into
one group. Fixed by comparing against the group's running average and only trusting normals that two methods agree on.
**Result:** the chair splits into its real parts (normals within 12-21 deg per part). Checked against the photos: the
blob grain matches the photo texture direction to a median of **9 deg**, where a random guess gives 44 deg.

![Figure 3.1: the chair split into surfaces, one colour per surface](figures/2026-10-12_chair_shadow_tests/s3_01_chair_clusters.jpg)
*Figure 3.1: the chair split into surfaces, one colour per surface*

![Figure 3.2: grain vs the real photo (green = within 20 deg)](figures/2026-10-12_chair_shadow_tests/s3_02_grain_vs_photo_interior.jpg)
*Figure 3.2: grain vs the real photo (green = within 20 deg)*

## Stage 4: Which blobs are in shadow (done, with additions)

**4.1 Sun-blocked test (as in the brief).** For each blob, look toward the sun: is any other part of the chair in the
way? This works like a shadow map in computer graphics. **[Addition]** A surface facing *away* from the sun is also
dark, even if nothing blocks it ("attached shadow"). The brief's blocked-test misses that, so a facing test was added.
It needs to know which side of each surface faces out, which comes from the cameras that saw it. Result: the
predicted shadow on the grass matches the photos (98% caught). On the chair it is right on the legs, under the seat
and behind the backrest, but over-predicts on the backrest front, where the sun hits almost edge-on.

**4.2 [Addition] Shadow from photo brightness.** Look up each chair blob in every photo (about 128 per blob) and take
its typical brightness. Shaded plastic is about 13 times darker than sunlit plastic, the same sun/sky ratio measured on
the grass, so the two groups separate cleanly. Result: 45% of the chair is in shadow. This method shares nothing with
4.1, and they agree well (precision 0.76).

**4.3 [Addition] Combined labels.** Brightness decides, except where the sun hits almost edge-on (16% of blobs). There
"weak sun" and "shade" look alike, and the geometry from 4.1 decides.

**4.4 [Addition] Retraining the shadow detector.** SDDNet missed most of the chair's shadow (it called 18% shadowed;
the dark navy plastic fools it). It was fine-tuned for 2.5 minutes using the 3D labels projected back into the photos,
and checked on frames held out from training. Its agreement with the sun geometry on those frames rose from 0.21 to
0.48, so it learned something physically consistent. Caveat: one scene, so this shows it can learn, not that it works
everywhere.

**4.5 Adjacent brightness and colour.** A blob is in shadow if it is much darker (under 0.3x) than
the lit blobs around it, within about 30 cm, on the same chair, *and* its colour still matches theirs, so dark paint
isn't mistaken for shadow. Finding: shaded plastic is measurably bluer (lit only by blue sky), a real shadow signature.
Result: 44.5% of the chair in shadow, consistent with 4.2, and it should do better on multi-coloured objects.

![Figure 4.1: sun-geometry prediction, front view (red shadow, cyan lit)](figures/2026-10-12_chair_shadow_tests/s4_01_geometry_front.jpg)
*Figure 4.1: sun-geometry prediction, front view (red shadow, cyan lit)*

![Figure 4.2: chair brightness splits into a sunlit and a shaded group](figures/2026-10-12_chair_shadow_tests/s4_04_brightness_histogram.jpg)
*Figure 4.2: chair brightness splits into a sunlit and a shaded group*

![Figure 4.4: retrained detector vs original on a frame it never saw](figures/2026-10-12_chair_shadow_tests/s4_11_retrained_detector_649.jpg)
*Figure 4.4: retrained detector vs original on a frame it never saw*

![Figure 4.5: adjacent brightness-and-colour detection, front view](figures/2026-10-12_chair_shadow_tests/s4_07_user_adjacent_method_front.jpg)
*Figure 4.5: adjacent brightness-and-colour detection, front view*

## Stage 5: Repainting the shadows (next, not started)

**What:** for each shadowed blob, take colour and brightness from lit blobs of the same surface (stage 3 group),
preferring those along the grain direction. The brief's rule: same surface and lit, *not* simply the brightest nearby,
to avoid pulling colour from other materials or highlights. Planned refinement: lift each blob toward its lit
neighbours' level rather than overwrite it, so its own small detail survives.

## Glossary

- **Gaussian / splat:** one soft coloured blob; the whole scene is millions of them.
- **Structure from Motion (SfM), COLMAP:** recovers camera positions and a rough 3D point cloud from photos.
- **Normal:** the direction a surface faces.
- **Cast shadow / attached shadow:** shadow from something blocking the sun / a surface turned away from the sun.
- **SDDNet:** an existing neural network that marks shadow pixels in a photo.
- **IoU (intersection over union):** overlap score between two regions; 1 = identical, 0 = no overlap.
- **Precision:** of the blobs a method calls shadow, the share that the other method also calls shadow.
- **Held-out frames:** photos kept out of training, used only to test.
