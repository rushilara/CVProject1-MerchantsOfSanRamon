"""Stage 1 with claim maps.

Besides the near-best locations of each patch, keep a coarse map of the best
correlation over all rotations and scales at every sky position. Stage 2 can
then ask "how well does patch p match at (x, y)?" for any location, which is
what verifying every constellation drawing needs.
"""

from __future__ import annotations

import pickle
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from q5_search import ANGLES, SCALES, KEEP, MERGE_PX, PER_ANGLE, _subpixel, load_gray, rotate_scale

HERE = Path(__file__).resolve().parent
ROOT = HERE / "participant"
CACHE = HERE / "cache_v3"
POOL = 4


def search_patch(sky, query):
    query = np.ascontiguousarray(query, dtype=np.float32)
    found = []
    best_map = None
    kernel = np.ones((15, 15), np.float32)
    for scale in SCALES:
        for angle in ANGLES:
            response = cv2.matchTemplate(sky, rotate_scale(query, angle, scale), cv2.TM_CCOEFF_NORMED)
            best_map = response if best_map is None else np.maximum(best_map, response, out=best_map)
            peak = float(response.max())
            if peak < 0.35:
                continue
            dilated = cv2.dilate(response, kernel)
            ys, xs = np.nonzero((response == dilated) & (response >= peak - 0.08))
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
    h, w = best_map.shape
    ph, pw = -(-h // POOL) * POOL, -(-w // POOL) * POOL
    padded = np.full((ph, pw), -1.0, np.float32)
    padded[:h, :w] = best_map
    pooled = padded.reshape(ph // POOL, POOL, pw // POOL, POOL).max(axis=(1, 3))
    return kept, pooled.astype(np.float16)


_SKY = None


def _init(sky_path):
    global _SKY
    cv2.setNumThreads(1)
    _SKY = load_gray(sky_path).astype(np.float32)


def _work(query):
    return search_patch(_SKY, query)


def search_folder(sky_path, patch_paths, tag, workers=5):
    hyp_path = CACHE / f"{tag}.pkl"
    map_path = CACHE / f"{tag}_maps.npy"
    if hyp_path.exists() and map_path.exists():
        with hyp_path.open("rb") as handle:
            names, hyps = pickle.load(handle)
        return names, hyps, np.load(map_path, mmap_mode="r")
    names = [Path(p).stem for p in patch_paths]
    images = [load_gray(p) for p in patch_paths]
    with ProcessPoolExecutor(max_workers=workers, initializer=_init, initargs=(str(sky_path),)) as pool:
        out = list(pool.map(_work, images, chunksize=1))
    hyps = [o[0] for o in out]
    maps = np.stack([o[1] for o in out])
    CACHE.mkdir(exist_ok=True)
    np.save(map_path, maps)
    with hyp_path.open("wb") as handle:
        pickle.dump((names, hyps), handle)
    return names, hyps, np.load(map_path, mmap_mode="r")


def search_scene(scene_dir, tag, workers=5):
    scene_dir = Path(scene_dir)
    sky_path = scene_dir / f"{scene_dir.name}_image.png"
    patch_paths = sorted((scene_dir / "patches").glob("patch_*.png"))
    return search_folder(sky_path, patch_paths, tag, workers)


def real_scene_dirs():
    dirs = [("train_" + d.name, d) for d in sorted((ROOT / "train").iterdir()) if d.is_dir()]
    dirs += [("val_" + d.name, d) for d in sorted((ROOT / "validation").iterdir()) if d.is_dir()]
    return dirs


def synthetic_scene_dirs():
    root = HERE / "synthetic"
    if not root.exists():
        return []
    return [("syn_" + d.name, d) for d in sorted(root.iterdir()) if d.is_dir()]


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "real"
    dirs = {"real": real_scene_dirs, "synthetic": synthetic_scene_dirs}[which]()
    prefix = sys.argv[2] if len(sys.argv) > 2 else ""
    for tag, scene_dir in dirs:
        if prefix and not tag.startswith(prefix):
            continue
        print(f"searching {tag}", flush=True)
        search_scene(scene_dir, tag)
        print(f"done {tag}", flush=True)
