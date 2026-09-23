"""Stage 1: rotation/scale sweep of normalised cross-correlation.

Keeps every near-best sky location per patch. The sky contains planted copies
of figure-star neighbourhoods, and bright stars look alike, so the correct
location is often tied with others. Stage 2 chooses among them.
"""

from __future__ import annotations

import pickle
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent / "participant"
CACHE = Path(__file__).resolve().parent / "cache_v2"

ANGLES = tuple(range(0, 360, 10))
SCALES = (0.88, 1.0, 1.14)
MERGE_PX = 12.0
KEEP = 16
PER_ANGLE = 4


def load_gray(path):
    return np.array(Image.open(path).convert("L"))


def rotate_scale(query, angle, scale):
    matrix = cv2.getRotationMatrix2D((15.5, 15.5), float(angle), float(scale))
    return cv2.warpAffine(
        query, matrix, (32, 32),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=float(np.median(query)),
    )


def _parabola(left, center, right):
    denom = left - 2.0 * center + right
    if abs(denom) < 1e-8:
        return 0.0
    return float(np.clip(0.5 * (left - right) / denom, -0.75, 0.75))


def _subpixel(response, px, py):
    h, w = response.shape
    dx = _parabola(response[py, px - 1], response[py, px], response[py, px + 1]) if 0 < px < w - 1 else 0.0
    dy = _parabola(response[py - 1, px], response[py, px], response[py + 1, px]) if 0 < py < h - 1 else 0.0
    return px + 16.0 + dx, py + 16.0 + dy


def search_patch(sky, query):
    """Up to KEEP distinct (ncc, x, y, angle, scale), best first."""
    query = np.ascontiguousarray(query, dtype=np.float32)
    found = []
    kernel = np.ones((15, 15), np.float32)
    for scale in SCALES:
        for angle in ANGLES:
            response = cv2.matchTemplate(sky, rotate_scale(query, angle, scale), cv2.TM_CCOEFF_NORMED)
            peak = float(response.max())
            if peak < 0.35:
                continue
            dilated = cv2.dilate(response, kernel)
            ys, xs = np.nonzero((response == dilated) & (response >= peak - 0.08))
            if len(xs) == 0:
                continue
            for index in np.argsort(-response[ys, xs])[:PER_ANGLE]:
                px, py = int(xs[index]), int(ys[index])
                x, y = _subpixel(response, px, py)
                found.append((float(response[py, px]), float(x), float(y), int(angle), float(scale)))
    found.sort(reverse=True)
    kept = []
    for hyp in found:
        if all(np.hypot(hyp[1] - k[1], hyp[2] - k[2]) > MERGE_PX for k in kept):
            kept.append(hyp)
        if len(kept) >= KEEP:
            break
    return kept


_SKY = None


def _init(sky_path):
    global _SKY
    cv2.setNumThreads(1)
    _SKY = load_gray(sky_path).astype(np.float32)


def _work(query):
    return search_patch(_SKY, query)


def search_scene(scene_dir, workers=7, tag=None):
    scene_dir = Path(scene_dir)
    tag = tag or scene_dir.name
    cache_path = CACHE / f"{tag}.pkl"
    if cache_path.exists():
        with cache_path.open("rb") as handle:
            return pickle.load(handle)
    sky_path = scene_dir / f"{scene_dir.name}_image.png"
    paths = sorted((scene_dir / "patches").glob("patch_*.png"))
    names = [p.stem for p in paths]
    images = [load_gray(p) for p in paths]
    with ProcessPoolExecutor(max_workers=workers, initializer=_init, initargs=(str(sky_path),)) as pool:
        hyps = list(pool.map(_work, images, chunksize=1))
    CACHE.mkdir(exist_ok=True)
    with cache_path.open("wb") as handle:
        pickle.dump((names, hyps), handle)
    return names, hyps


def all_scene_dirs():
    dirs = [("train_" + d.name, d) for d in sorted((ROOT / "train").iterdir()) if d.is_dir()]
    dirs += [("val_" + d.name, d) for d in sorted((ROOT / "validation").iterdir()) if d.is_dir()]
    return dirs


if __name__ == "__main__":
    only = sys.argv[1] if len(sys.argv) > 1 else None
    for tag, scene_dir in all_scene_dirs():
        if only and not tag.startswith(only):
            continue
        print(f"searching {tag}", flush=True)
        search_scene(scene_dir, tag=tag)
        print(f"done {tag}", flush=True)
