# Paper

**Shadow-Free Gaussian Splatting: Sun Estimation from Cast-Shadow Geometry and Surface-Aware Inpainting of 3D Gaussian Splats**

Bhanu Singh, 2026

- [Preprint PDF](shadow_free_gaussian_splatting_preprint.pdf) (34 pages)
- Status: preprint, in preparation for arXiv (cs.CV, cross-list cs.GR). This page will link the arXiv entry once it is posted.
- This is an ongoing study; results on further scenes will be added in later versions.

## Abstract

A 3D Gaussian splat trained from photos taken under one sun stores the cast shadows of that moment in the colours of its Gaussians, so the shadows travel with the model when it is moved or relit. We present a training-free method that removes these baked-in shadows from a finished splat reconstructed from a single phone video. Rather than decomposing the scene into materials and lighting, we treat shadow removal as surface-aware inpainting in 3D. The pipeline estimates the sun direction from the geometry of the object's cast shadow by orthographic shadow mapping and boundary chamfer matching, groups Gaussians into surfaces and measures each surface's texture direction (grain) with a structure tensor, labels every Gaussian as sunlit, attached shadow or cast shadow, and repaints shadowed Gaussians from sunlit Gaussians of the same surface, searched along the grain. On the ground, a clone-stamp procedure replaces shadowed Gaussians with translated copies of nearby sunlit ones, carrying both their colour and their geometry. On a chair scene the sun elevation is recovered to within 0.43 degrees of the value computed from GPS and capture time. On a painted bear figurine it comes out 4.98 degrees high, and the object-side stages fail on multi-coloured, curved surfaces, for reasons we analyse. The ground shadow on concrete, which a per-Gaussian gain lifted only to 63% of the surrounding sunlit brightness, reaches 97% with the clone stamp, with fine-detail contrast restored to 87-101% of the surrounding concrete. We show that the residual smudge of colour-only fills comes from geometry: Gaussians trained on the dark shadow are 1.7 times larger than those on sunlit concrete. These are the first two scenes of an ongoing study; results on further scenes will be added in later versions.

## Citation

```bibtex
@misc{singh2026shadowfree,
  author = {Singh, Bhanu},
  title  = {Shadow-Free Gaussian Splatting: Sun Estimation from Cast-Shadow Geometry and Surface-Aware Inpainting of 3D Gaussian Splats},
  year   = {2026},
  note   = {Preprint},
  url    = {https://github.com/thebhanusingh/Claude_Test}
}
```
