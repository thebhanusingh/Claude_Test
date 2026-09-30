#!/usr/bin/env python
"""Extract the sharpest frame from each of N evenly sized windows of a video.

Handheld video has motion blur on many frames; picking the sharpest frame per window
(by variance of the Laplacian) gives COLMAP and splat training much crisper input
than ffmpeg's evenly spaced extraction.

Usage: select_sharp_frames.py <video> <out_dir> [num_frames]
"""
import sys
from pathlib import Path

import cv2
import numpy as np

video, out_dir = sys.argv[1], Path(sys.argv[2])
num_frames = int(sys.argv[3]) if len(sys.argv) > 3 else 650
out_dir.mkdir(parents=True, exist_ok=True)

# Pass 1: sharpness score for every frame (on a downscaled grayscale copy).
cap = cv2.VideoCapture(video)
scores = []
while True:
    ok, frame = cap.read()
    if not ok:
        break
    gray = cv2.cvtColor(cv2.resize(frame, (960, 540)), cv2.COLOR_BGR2GRAY)
    scores.append(cv2.Laplacian(gray, cv2.CV_64F).var())
cap.release()
scores = np.array(scores)
total = len(scores)
if total == 0:
    sys.exit(f"ERROR: could not decode any frames from {video}")

# Pick the sharpest frame in each window.
edges = np.linspace(0, total, min(num_frames, total) + 1).astype(int)
picks = {int(a + np.argmax(scores[a:b])) for a, b in zip(edges[:-1], edges[1:]) if b > a}

# Pass 2: write the picked frames at full resolution.
cap = cv2.VideoCapture(video)
idx = written = 0
while True:
    ok, frame = cap.read()
    if not ok:
        break
    if idx in picks:
        written += 1
        cv2.imwrite(str(out_dir / f"frame_{written:05d}.png"), frame)
    idx += 1
cap.release()

chosen = scores[sorted(picks)]
print(f"{total} frames scanned, wrote {written} sharpest frames to {out_dir}")
print(f"sharpness: all frames median {np.median(scores):.1f}, chosen median {np.median(chosen):.1f}")
