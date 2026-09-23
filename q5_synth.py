"""Synthetic labelled scenes built from the real validation skies.

Only three labelled scenes exist, all with large constellations. These scenes
place a random pattern (uniform over all drawings, so small ones are tested)
onto real bright stars of a real sky, issue degraded patches of a subset of
its nodes, add off-figure stars (sometimes shaped like a decoy drawing), and
add absent patches cut from other skies.
"""

from __future__ import annotations

import io
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from q5_identify import load_patterns

HERE = Path(__file__).resolve().parent
ROOT = HERE / "participant"
OUT = HERE / "synthetic"


def load_gray(path):
    return np.array(Image.open(path).convert("L"))


def star_catalog(sky):
    g = sky.astype(np.float32)
    dog = cv2.GaussianBlur(g, (0, 0), 1.2) - cv2.GaussianBlur(g, (0, 0), 5.0)
    dil = cv2.dilate(dog, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    peaks = (dog >= dil - 1e-4) & (dog > 6)
    b = 40
    peaks[:b] = peaks[-b:] = False
    peaks[:, :b] = peaks[:, -b:] = False
    ys, xs = np.nonzero(peaks)
    strength = dog[ys, xs]
    order = np.argsort(-strength)
    return np.column_stack([xs, ys])[order].astype(np.float32), strength[order]


def degrade(sky, x, y, rng):
    """Rotate, rescale, sub-pixel shift, blur, illumination drift, noise, JPEG."""
    angle = rng.uniform(0, 360)
    scale = rng.uniform(0.86, 1.16)
    dx, dy = rng.uniform(-0.5, 0.5, 2)
    big = sky[int(y) - 40:int(y) + 40, int(x) - 40:int(x) + 40].astype(np.float32)
    center = (40.0 + (x - int(x)), 40.0 + (y - int(y)))
    M = cv2.getRotationMatrix2D(center, angle, scale)
    M[0, 2] += 15.5 - center[0] + dx
    M[1, 2] += 15.5 - center[1] + dy
    patch = cv2.warpAffine(big, M, (32, 32), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    patch = cv2.GaussianBlur(patch, (0, 0), rng.uniform(0.3, 0.7))
    yy, xx = np.mgrid[:32, :32] / 31.0
    gain = rng.uniform(0.85, 1.15) + rng.uniform(-0.08, 0.08) * xx + rng.uniform(-0.08, 0.08) * yy
    patch = patch * gain + rng.uniform(-10, 10)
    patch = patch + rng.normal(0, rng.uniform(1.0, 3.0), patch.shape)
    patch = np.clip(patch, 0, 255).astype(np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(patch).save(buffer, format="JPEG", quality=int(rng.uniform(80, 95)))
    return np.array(Image.open(io.BytesIO(buffer.getvalue())).convert("L"))


def place_pattern(nodes, catalog, strength, rng, extent_frac):
    nodes = nodes.astype(np.float64).copy()
    nodes -= nodes.mean(0)
    if rng.random() < 0.5:
        nodes[:, 0] *= -1
    aspect = rng.uniform(0.85, 1.15)
    nodes[:, 0] *= aspect
    theta = rng.uniform(0, 2 * np.pi)
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    nodes = nodes @ R.T
    span = max(np.ptp(nodes[:, 0]), np.ptp(nodes[:, 1]), 1.0)
    nodes *= extent_frac * 3000 / span
    lo = -nodes.min(0) + 120
    hi = 3000 - nodes.max(0) - 120
    if np.any(hi <= lo):
        return None
    nodes += rng.uniform(lo, hi)
    # Catalog is sorted brightest first, so the first star in range is the brightest nearby.
    bright = catalog[:3000]
    snapped = []
    for p in nodes:
        d = np.hypot(bright[:, 0] - p[0], bright[:, 1] - p[1])
        near = np.flatnonzero(d <= 60)
        snapped.append(bright[near[0]] if len(near) else None)
    return snapped


def make_scene(index, sky_path, other_skies, patterns, rng):
    sky = load_gray(sky_path)
    catalog, strength = star_catalog(sky)
    names = list(patterns)
    for _ in range(50):
        figure_name = names[rng.integers(len(names))]
        snapped = place_pattern(patterns[figure_name], catalog, strength, rng, rng.uniform(0.3, 0.75))
        if snapped is None:
            continue
        stars = [s for s in snapped if s is not None]
        unique = {tuple(s) for s in stars}
        if len(unique) >= min(3, len(patterns[figure_name])):
            break
    figure = [np.array(s) for s in unique]
    rng.shuffle(figure)
    n_issue = max(min(len(figure), 3), int(round(rng.uniform(0.5, 0.9) * len(figure))))
    figure = figure[:n_issue]

    taken = {tuple(s) for s in figure}
    off = []
    if rng.random() < 0.35:
        decoy_name = names[rng.integers(len(names))]
        if decoy_name != figure_name:
            snapped = place_pattern(patterns[decoy_name], catalog, strength, rng, rng.uniform(0.3, 0.7))
            if snapped:
                for s in snapped:
                    if s is not None and tuple(s) not in taken:
                        taken.add(tuple(s))
                        off.append(s)
    n_off = int(rng.uniform(1.0, 2.5) * len(figure)) + int(rng.integers(0, 15))
    bright = catalog[:1000]
    while len(off) < n_off:
        pool = bright if rng.random() < 0.5 else catalog
        s = pool[rng.integers(len(pool))]
        if tuple(s) not in taken and all(np.hypot(*(s - t)) > 20 for t in figure):
            taken.add(tuple(s))
            off.append(s)
    n_present = len(figure) + len(off)
    n_absent = int(round(n_present * rng.uniform(0.35, 0.8)))

    items = [("fig", s) for s in figure] + [("off", s) for s in off]
    for _ in range(n_absent):
        other = load_gray(other_skies[rng.integers(len(other_skies))])
        cat, _ = star_catalog(other)
        pool = cat[:1000] if rng.random() < 0.5 else cat
        items.append(("abs", (other, pool[rng.integers(len(pool))])))
    order = rng.permutation(len(items))
    items = [items[i] for i in order][:87]

    scene = f"syn_{index:02d}"
    folder = OUT / scene
    (folder / "patches").mkdir(parents=True, exist_ok=True)
    link = folder / f"{scene}_image.png"
    if link.exists() or link.is_symlink():
        link.unlink()
    os.symlink(os.path.relpath(sky_path, folder), link)
    truth = {}
    for k, (kind, payload) in enumerate(items, start=1):
        name = f"patch_{k:02d}"
        if kind == "abs":
            other, s = payload
            patch = degrade(other, s[0], s[1], rng)
            truth[name] = None
        else:
            patch = degrade(sky, payload[0], payload[1], rng)
            truth[name] = [int(payload[0]), int(payload[1]), 1 if kind == "fig" else 0]
        Image.fromarray(patch).save(folder / "patches" / f"{name}.png")
    with (folder / "truth.json").open("w") as handle:
        json.dump({"constellation": figure_name, "patches": truth}, handle)
    return scene, figure_name, len(items), len(figure)


def main(count=12, seed=7):
    patterns = load_patterns(ROOT / "patterns")
    val_skies = sorted((ROOT / "validation").glob("*/*_image.png"))
    all_skies = val_skies + sorted((ROOT / "train").glob("*/*_image.png"))
    rng = np.random.default_rng(seed)
    for i in range(count):
        sky_path = val_skies[i % len(val_skies)]
        others = [s for s in all_skies if s != sky_path]
        scene, name, n, nf = make_scene(i + 1, sky_path, others, patterns, rng)
        print(f"{scene}: {name:18} patches {n:2d} figure issued {nf}", flush=True)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 12)
