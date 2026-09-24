"""HPC-scale constellation pipeline.

The 0.306 submission failed because figure placement was proposed only from
NCC peaks. Bright-star patches correlate almost equally at many decoys, so
the true geometry is often never proposed, large drawings win by offering
more targets, and only the top five drawings were verified.

This version spends compute on the things that actually move the Kaggle
score (55% of it is the name + figure recovery):

1. Detect bright stars independently of the patches. Align every one of
   the 48 drawings to those stars (similarity + reflection).
2. Also propose transforms from near-tied patch peaks (the v3 idea).
3. Verify every drawing with per-patch claim maps and a Hungarian
   node/patch assignment. Rank by chance-corrected high-claim excess so
   Hydra/Eridanus cannot win just by being large.
4. When the winner is clear, spare-fill leftover nodes with leftover
   patches. Figure recovery scores predicted points against true figure
   stars, not patch identity, so a point on every node is worth 25%.
5. On HPC (Q5_FINE=1) re-run stage 1 on a 5° × 7-scale grid with 2×
   claim maps. Existing cache_v3 is used otherwise so a laptop can still
   produce a submission.

Environment:
  Q5_WORKERS   process count for the correlation sweep (default: CPU count)
  Q5_FINE      1 = finer search written to cache_hpc/
  Q5_CACHE     override cache directory
"""

from __future__ import annotations

import json
import hashlib
import os
import pickle
import sys
import time
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from scipy.ndimage import maximum_filter
from scipy.optimize import linear_sum_assignment
from scipy.stats import theilslopes

from q5_constellation import format_row, parse_truth, score_scene
from q5_identify import Registrar, SkySupport, load_patterns
from q5_search import _subpixel, load_gray, rotate_scale
from q5_synth import star_catalog

HERE = Path(__file__).resolve().parent
ROOT = HERE / "participant"

FINE = os.environ.get("Q5_FINE", "0") == "1"
if FINE:
    ANGLES = tuple(range(0, 360, 5))
    SCALES = (0.82, 0.88, 0.94, 1.00, 1.07, 1.14, 1.22)
    DEFAULT_POOL = 2
    DEFAULT_CACHE = HERE / "cache_hpc"
else:
    ANGLES = tuple(range(0, 360, 10))
    SCALES = (0.88, 1.00, 1.14)
    DEFAULT_POOL = 4
    DEFAULT_CACHE = HERE / "cache_v3"

CACHE = Path(os.environ.get("Q5_CACHE", DEFAULT_CACHE))
WORKERS = int(os.environ.get("Q5_WORKERS", os.cpu_count() or 4))
KEEP = 24
PER_ANGLE = 6
MERGE_PX = 12.0

PARAMS = dict(
    tie_gap=0.04,
    max_alternatives=12,
    min_ncc_for_fit=0.50,
    bright_pct=0.75,
    min_base_px=50.0,
    max_stars=120,
    max_star_bases=800,
    max_node_pairs=24,
    keep_transforms=24,
    claim_radius=2,
    claim_min=0.68,
    quick_level=0.82,
    rule="excess",
    select="residual",
    high=0.88,
    present_ncc=0.90,
    moderate_ncc=0.55,
    moderate_margin=0.04,
    spare_fill=True,
    spare_min=0.62,
    min_figure_claims=3,
    win_margin=0.4,
)


# ---------------------------------------------------------------------------
# Stage 1 — correlation sweep + claim maps
# ---------------------------------------------------------------------------

def search_patch(sky, query, angles=ANGLES, scales=SCALES, pool=DEFAULT_POOL):
    query = np.ascontiguousarray(query, dtype=np.float32)
    found = []
    best_map = None
    kernel = np.ones((15, 15), np.float32)
    for scale in scales:
        for angle in angles:
            response = cv2.matchTemplate(sky, rotate_scale(query, angle, scale), cv2.TM_CCOEFF_NORMED)
            best_map = response if best_map is None else np.maximum(best_map, response, out=best_map)
            peak = float(response.max())
            if peak < 0.32:
                continue
            dilated = cv2.dilate(response, kernel)
            ys, xs = np.nonzero((response == dilated) & (response >= peak - 0.10))
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
    ph, pw = -(-h // pool) * pool, -(-w // pool) * pool
    padded = np.full((ph, pw), -1.0, np.float32)
    padded[:h, :w] = best_map
    pooled = padded.reshape(ph // pool, pool, pw // pool, pool).max(axis=(1, 3))
    return kept, pooled.astype(np.float16)


_SKY = None
_ANGLES = ANGLES
_SCALES = SCALES
_POOL = DEFAULT_POOL


def _init(sky_path, angles, scales, pool):
    global _SKY, _ANGLES, _SCALES, _POOL
    cv2.setNumThreads(1)
    _SKY = load_gray(sky_path).astype(np.float32)
    _ANGLES = angles
    _SCALES = scales
    _POOL = pool


def _work(query):
    return search_patch(_SKY, query, _ANGLES, _SCALES, _POOL)


def infer_pool(maps, sky_shape):
    resp_h = max(int(sky_shape[0]) - 31, 1)
    return max(1, int(round(resp_h / maps.shape[1])))


def resolve_cache(tag):
    # An explicit cache directory is authoritative. Never silently mix it
    # with older local caches (especially on a cluster).
    folders = (CACHE,) if "Q5_CACHE" in os.environ else (CACHE, HERE / "cache_hpc", HERE / "cache_v3")
    for folder in folders:
        if (folder / f"{tag}.pkl").exists() and (folder / f"{tag}_maps.npy").exists():
            return folder
    return None


def search_folder(sky_path, patch_paths, tag, workers=None, force=False):
    workers = workers if workers is not None else max(1, WORKERS)
    cached = None if force else resolve_cache(tag)
    if cached is not None:
        with (cached / f"{tag}.pkl").open("rb") as handle:
            names, hyps = pickle.load(handle)
        maps = np.load(cached / f"{tag}_maps.npy", mmap_mode="r")
        return names, hyps, maps
    names = [Path(p).stem for p in patch_paths]
    images = [load_gray(p) for p in patch_paths]
    CACHE.mkdir(parents=True, exist_ok=True)
    if workers <= 1:
        sky = load_gray(sky_path).astype(np.float32)
        out = [search_patch(sky, q) for q in images]
    else:
        # spawn, not fork: the parent has already imported cv2, and forking
        # OpenCV workers is a known hang. The last job sat 8 hours on the
        # first scene at ~300 MB RSS, i.e. the pool never ran.
        workers = min(workers, len(images))
        print(f"  {len(images)} patches, {workers} spawn workers", flush=True)
        ctx = mp.get_context("spawn")
        out = [None] * len(images)
        done = 0
        t0 = time.time()
        with ProcessPoolExecutor(
            max_workers=workers,
            mp_context=ctx,
            initializer=_init,
            initargs=(str(sky_path), ANGLES, SCALES, DEFAULT_POOL),
        ) as pool:
            futs = {pool.submit(_work, image): i for i, image in enumerate(images)}
            for fut in as_completed(futs):
                index = futs[fut]
                out[index] = fut.result()
                done += 1
                print(f"  {tag} {done}/{len(images)} {time.time() - t0:.0f}s", flush=True)
    hyps = [o[0] for o in out]
    maps = np.stack([o[1] for o in out])
    # Write the map first, the hypothesis file last. A preempted job then
    # resumes from the last scene that has both files.
    partial = CACHE / f".{tag}_maps.partial.npy"
    np.save(partial, maps)
    os.replace(partial, CACHE / f"{tag}_maps.npy")
    with (CACHE / f"{tag}.pkl").open("wb") as handle:
        pickle.dump((names, hyps), handle)
    return names, hyps, np.load(CACHE / f"{tag}_maps.npy", mmap_mode="r")


def search_scene(scene_dir, tag, workers=None, force=False):
    scene_dir = Path(scene_dir)
    sky_path = scene_dir / f"{scene_dir.name}_image.png"
    patch_paths = sorted((scene_dir / "patches").glob("patch_*.png"))
    return search_folder(sky_path, patch_paths, tag, workers, force)


# ---------------------------------------------------------------------------
# Stage 2 — propose from stars + patches, verify every drawing
# ---------------------------------------------------------------------------

class Claims:
    def __init__(self, maps, radius, pool):
        self.pool = pool
        self.pad = 16.0 + 0.5 * (pool - 1)
        # Keep only one float32 plane live, rather than a list and stacked
        # copy of every plane. The float16 values are identical.
        self.dil = np.empty(maps.shape, dtype=np.float16)
        for index, m in enumerate(maps):
            self.dil[index] = maximum_filter(
                np.asarray(m, dtype=np.float32), size=2 * radius + 1, mode="nearest"
            )
        self.best = self.dil.max(axis=0).astype(np.float32)
        self.shape = self.best.shape

    def cells(self, xy):
        xy = np.asarray(xy, np.float64)
        j = np.round((xy[..., 0] - self.pad) / self.pool).astype(np.int64)
        i = np.round((xy[..., 1] - self.pad) / self.pool).astype(np.int64)
        inside = (i >= 0) & (i < self.shape[0]) & (j >= 0) & (j < self.shape[1])
        return np.clip(i, 0, self.shape[0] - 1), np.clip(j, 0, self.shape[1] - 1), inside

    def best_at(self, xy):
        i, j, inside = self.cells(xy)
        return np.where(inside, self.best[i, j], -1.0)

    def matrix(self, xy):
        i, j, inside = self.cells(xy)
        m = self.dil[:, i, j].T.astype(np.float32)
        m[~inside] = -1.0
        return m


def candidate_cloud(hyps, support, p):
    pts, owners = [], []
    for index, hs in enumerate(hyps):
        if not hs or hs[0][0] < p["min_ncc_for_fit"]:
            continue
        for h in hs[: p["max_alternatives"]]:
            if h[0] < hs[0][0] - p["tie_gap"]:
                continue
            pts.append((h[1], h[2]))
            owners.append(index)
    pts = np.asarray(pts, np.float64).reshape(-1, 2)
    owners = np.asarray(owners)
    if len(pts):
        pct, _ = support.percentile(pts)
        keep = pct >= p["bright_pct"]
        pts, owners = pts[keep], owners[keep]
    return pts, owners


def bright_stars(sky, p):
    catalog, _ = star_catalog(sky)
    if len(catalog) == 0:
        return np.zeros((0, 2), np.float64)
    return catalog[: p["max_stars"]].astype(np.float64)


def _pair_proposals(nodes, pts, base_a, base_b, claims, p, rng):
    n = len(nodes)
    if n < 3 or len(base_a) == 0:
        return []
    ii, jj = np.nonzero(~np.eye(n, dtype=bool))
    sep = np.hypot(*(nodes[jj] - nodes[ii]).T)
    span = max(np.ptp(nodes[:, 0]), np.ptp(nodes[:, 1]), 1.0)
    good = np.flatnonzero(sep >= 0.12 * span)
    if len(good) > p["max_node_pairs"]:
        good = rng.choice(good, p["max_node_pairs"], replace=False)
    qa, qb = pts[base_a], pts[base_b]
    vq = qb - qa
    lq = np.hypot(vq[:, 0], vq[:, 1])
    angq = np.arctan2(vq[:, 1], vq[:, 0])
    proposals = []
    for reflect in (False, True):
        src = nodes.astype(np.float64).copy()
        if reflect:
            src[:, 0] = -src[:, 0]
        for g in good:
            i, j = int(ii[g]), int(jj[g])
            vp = src[j] - src[i]
            scale = lq / (np.hypot(*vp) + 1e-9)
            ang = angq - np.arctan2(vp[1], vp[0])
            c, s = np.cos(ang) * scale, np.sin(ang) * scale
            rel = src - src[i]
            px = qa[:, 0][:, None] + c[:, None] * rel[:, 0] - s[:, None] * rel[:, 1]
            py = qa[:, 1][:, None] + s[:, None] * rel[:, 0] + c[:, None] * rel[:, 1]
            extent = np.ptp(px, axis=1) + np.ptp(py, axis=1)
            pred = np.stack([px, py], axis=-1)
            quick = (claims.best_at(pred) >= p["quick_level"]).sum(axis=1).astype(np.float64)
            quick[extent < 350] = -1
            for t in np.argsort(-quick)[: p["keep_transforms"]]:
                if quick[t] <= 0:
                    break
                proposals.append((quick[t], pred[t]))
    proposals.sort(key=lambda r: -r[0])
    return [pred for _, pred in proposals[: p["keep_transforms"]]]


def propose_from_patches(nodes, pts, owners, claims, p, rng):
    if len(pts) < 2:
        return []
    n = len(pts)
    a, b = np.nonzero(np.ones((n, n), bool))
    keep = owners[a] != owners[b]
    a, b = a[keep], b[keep]
    length = np.hypot(*(pts[b] - pts[a]).T)
    keep = length >= p["min_base_px"]
    a, b = a[keep], b[keep]
    if len(a) > 4000:
        pick = rng.choice(len(a), 4000, replace=False)
        a, b = a[pick], b[pick]
    return _pair_proposals(nodes, pts, a, b, claims, p, rng)


def propose_from_stars(nodes, stars, claims, p, rng):
    if len(stars) < 3:
        return []
    n = len(stars)
    a, b = np.triu_indices(n, 1)
    length = np.hypot(*(stars[b] - stars[a]).T)
    keep = length >= p["min_base_px"]
    a, b = a[keep], b[keep]
    if len(a) > p["max_star_bases"]:
        pick = rng.choice(len(a), p["max_star_bases"], replace=False)
        a, b = a[pick], b[pick]
    return _pair_proposals(nodes, stars, a, b, claims, p, rng)


def propose_from_hash(nodes, stars, support, k=6):
    if len(stars) < 3 or len(nodes) < 3:
        return []
    owners = np.arange(len(stars))
    reg = Registrar(stars, owners, np.ones(len(stars)), support, tol=18.0, lam=1.2)
    return reg.top_preds(nodes, k=k, top_bases=8, max_pairs=180)


def assign(pred, claims, p):
    m = claims.matrix(pred)
    cost = np.where(m >= p["claim_min"], -m, 10.0)
    rows, cols = linear_sum_assignment(cost)
    return [(int(r), int(c), float(m[r, c])) for r, c in zip(rows, cols) if m[r, c] >= p["claim_min"]]


def evidence(pairs, n_nodes, p, null_high):
    vals = np.array([v for _, _, v in pairs]) if pairs else np.zeros(0)
    n_high = float((vals >= p["high"]).sum())
    if p["rule"] == "mean":
        return float(vals.sum() / n_nodes)
    if p["rule"] == "excess":
        return n_high - null_high * n_nodes
    if p["rule"] == "blend":
        return n_high - null_high * n_nodes + float(vals.sum() / n_nodes)
    raise ValueError(p["rule"])


def null_high_rate(claims, support, p, rng, samples=4000):
    ys = rng.uniform(40, support.h - 40, samples)
    xs = rng.uniform(40, support.w - 40, samples)
    return float(np.mean(claims.best_at(np.column_stack([xs, ys])) >= p["high"]))


def scene_proposals(hyps, maps, sky, patterns, p=PARAMS, seed=0):
    rng = np.random.default_rng(seed)
    support = SkySupport(sky)
    pool = infer_pool(maps, sky.shape)
    claims = Claims(maps, p["claim_radius"], pool)
    pts, owners = candidate_cloud(hyps, support, p)
    stars = bright_stars(sky, p)
    star_reg = None
    if len(stars) >= 3:
        star_reg = Registrar(stars, np.arange(len(stars)), np.ones(len(stars)), support, tol=18.0, lam=1.2)
    proposals = {}
    names = [n for n, nodes in patterns.items() if len(nodes) >= 3]
    for i, name in enumerate(names, 1):
        nodes = patterns[name]
        preds = []
        if star_reg is not None:
            preds.extend(star_reg.top_preds(nodes, k=8, top_bases=8, max_pairs=180))
        preds.extend(propose_from_patches(nodes, pts, owners, claims, p, rng))
        # Exhaustive star-pair scoring is the expensive HPC extra.
        if FINE:
            preds.extend(propose_from_stars(nodes, stars, claims, p, rng))
        proposals[name] = preds
        if i == 1 or i == len(names) or i % 12 == 0:
            print(f"    drawings {i}/{len(names)}", flush=True)
    return dict(support=support, claims=claims, proposals=proposals, pool=pool)


def figure_stage(hyps, maps, sky, patterns, p=PARAMS, seed=0, cached=None):
    if cached is None:
        cached = scene_proposals(hyps, maps, sky, patterns, p, seed)
    proposals = cached["proposals"]
    claims = cached.get("claims")
    if claims is None:
        claims = Claims(maps, p["claim_radius"], cached.get("pool") or infer_pool(maps, sky.shape))
    support = cached.get("support") or SkySupport(sky)
    if not proposals:
        return "unknown", {}, []
    null_high = null_high_rate(claims, support, p, np.random.default_rng(seed))
    table = []
    for name, preds in proposals.items():
        nodes = patterns[name]
        best = None
        for pred in preds:
            pairs = assign(pred, claims, p)
            score = evidence(pairs, len(nodes), p, null_high)
            if best is None or score > best[0]:
                best = (score, pred, pairs)
        if best is not None:
            table.append((best[0], name, best[1], best[2]))
    if not table:
        return "unknown", {}, []
    if p.get("select") == "residual" and len(table) >= 8:
        sizes = np.array([len(patterns[r[1]]) for r in table], float)
        scores = np.array([r[0] for r in table], float)
        slope, intercept, _, _ = theilslopes(scores, sizes)
        table = [(s - (intercept + slope * n),) + r[1:] for s, n, r in zip(scores, sizes, table)]
    table.sort(key=lambda r: -r[0])
    score, name, pred, pairs = table[0]
    figure = {patch: (node, value) for node, patch, value in pairs}
    return name, {"pred": pred, "assigned": figure, "score": score, "claims": claims}, table


# ---------------------------------------------------------------------------
# Stage 3 — report (x, y, m)
# ---------------------------------------------------------------------------

def refine_location(sky_f, query, x, y, half=16):
    h, w = sky_f.shape
    xi, yi = int(round(x)), int(round(y))
    x0, y0 = max(0, xi - half - 16), max(0, yi - half - 16)
    x1, y1 = min(w, xi + half + 16), min(h, yi + half + 16)
    window = sky_f[y0:y1, x0:x1]
    if window.shape[0] < 33 or window.shape[1] < 33:
        return x, y, -1.0
    q = query.astype(np.float32)
    best = (-1.0, x, y)
    for s in SCALES:
        for a in ANGLES:
            res = cv2.matchTemplate(window, rotate_scale(q, a, s), cv2.TM_CCOEFF_NORMED)
            j = int(np.argmax(res))
            v = float(res.flat[j])
            if v > best[0]:
                py, px = divmod(j, res.shape[1])
                best = (v, x0 + px + 16.0, y0 + py + 16.0)
    return best[1], best[2], best[0]


def choose_present(hs, p):
    if not hs:
        return False
    best = hs[0]
    rival = next((h for h in hs[1:] if np.hypot(h[1] - best[1], h[2] - best[2]) > 22), None)
    margin = best[0] - (rival[0] if rival else 0.0)
    return best[0] >= p["present_ncc"] or (best[0] >= p["moderate_ncc"] and margin >= p["moderate_margin"])


def finalize(names, hyps, fig, sky, queries, p=PARAMS):
    sky_f = sky.astype(np.float32)
    results = {}
    assigned = fig.get("assigned", {}) if fig else {}
    pred = fig.get("pred") if fig else None
    claims = fig.get("claims") if fig else None
    table_margin = fig.get("margin", 1.0) if fig else 1.0
    confident = bool(assigned) and len(assigned) >= p["min_figure_claims"] and table_margin >= p["win_margin"]

    for index, name in enumerate(names):
        if index in assigned and pred is not None:
            node, _ = assigned[index]
            x, y, _ = refine_location(sky_f, queries[index], pred[node][0], pred[node][1])
            results[name] = (int(round(x)), int(round(y)), 1)
        elif choose_present(hyps[index], p):
            h = hyps[index][0]
            results[name] = (int(round(h[1])), int(round(h[2])), 0)
        else:
            results[name] = (-1, -1, -1)

    if p["spare_fill"] and confident and pred is not None and claims is not None:
        used_nodes = {node for node, _ in assigned.values()}
        unused_nodes = [i for i in range(len(pred)) if i not in used_nodes]
        unused = [i for i, n in enumerate(names) if results[n][2] == -1]
        if unused_nodes and unused:
            m = claims.matrix(pred)
            for ni in unused_nodes:
                best_p, best_v = None, p["spare_min"]
                for pi in unused:
                    if m[ni, pi] > best_v:
                        best_v, best_p = float(m[ni, pi]), pi
                if best_p is None:
                    continue
                x, y = pred[ni]
                if 16 <= x < sky.shape[1] - 16 and 16 <= y < sky.shape[0] - 16:
                    results[names[best_p]] = (int(round(x)), int(round(y)), 1)
                    unused.remove(best_p)
    return results


# ---------------------------------------------------------------------------
# I/O, evaluation, submission
# ---------------------------------------------------------------------------

def load_sky(scene_dir):
    scene_dir = Path(scene_dir)
    return np.array(Image.open(scene_dir / f"{scene_dir.name}_image.png").convert("L"))


def load_queries(scene_dir, names):
    return [np.array(Image.open(Path(scene_dir) / "patches" / f"{n}.png").convert("L")) for n in names]


def labelled_scenes(include_train=True, include_synthetic=True):
    scenes = []
    if include_train:
        gt = pd.read_csv(ROOT / "train_ground_truth.csv")
        for _, r in gt.iterrows():
            truth = {f"patch_{i:02d}": parse_truth(r[f"patch_{i:02d}"]) for i in range(1, int(r["n_patches"]) + 1)}
            scenes.append((f"train_{r['Id']}", ROOT / "train" / r["Id"], truth, r["constellation"]))
    if include_synthetic:
        root = HERE / "synthetic"
        if root.exists():
            for folder in sorted(root.iterdir()):
                if not folder.is_dir() or not (folder / "truth.json").exists():
                    continue
                info = json.loads((folder / "truth.json").read_text())
                truth = {k: (tuple(v) if v else None) for k, v in info["patches"].items()}
                scenes.append((f"syn_{folder.name}", folder, truth, info["constellation"]))
    return scenes


def validation_scenes():
    sample = pd.read_csv(ROOT / "sample_submission.csv")
    return [(f"val_{r['Id']}", ROOT / "validation" / r["Id"], int(r["n_patches"])) for _, r in sample.iterrows()]


def prepare(scenes, patterns, p=PARAMS, verbose=True, proposal_cache=None):
    out = {}
    for tag, folder, truth, true_name in scenes:
        if resolve_cache(tag) is None:
            print(f"{tag:22} SKIP (no cache)", flush=True)
            continue
        names, hyps, maps = search_scene(folder, tag)
        sky = load_sky(folder)
        cached = None
        if proposal_cache is not None:
            if tag not in proposal_cache:
                t = time.time()
                proposal_cache[tag] = {
                    "proposals": scene_proposals(hyps, maps, sky, patterns, p)["proposals"],
                    "pool": infer_pool(maps, sky.shape),
                }
                print(f"  proposals {tag} {time.time() - t:.0f}s", flush=True)
            cached = proposal_cache[tag]
        name, fig, table = figure_stage(hyps, maps, sky, patterns, p, cached=cached)
        if table:
            fig["margin"] = table[0][0] - (table[1][0] if len(table) > 1 else -10.0)
        out[tag] = dict(
            names=names, hyps=hyps, maps=maps, sky=sky, folder=folder,
            truth=truth, true_name=true_name, name=name, fig=fig, table=table,
        )
        if verbose:
            rank = next((k for k, r in enumerate(table) if r[1] == true_name), None)
            top = [(round(r[0], 2), r[1]) for r in table[:3]]
            print(f"{tag:22} true {true_name:16} pred {name:16} rank {rank} top {top}", flush=True)
    return out


def score_prepared(prepared, p=PARAMS, verbose=False):
    rows = []
    for tag, d in prepared.items():
        queries = load_queries(d["folder"], d["names"])
        res = finalize(d["names"], d["hyps"], d["fig"], d["sky"], queries, p)
        m = score_scene(d["truth"], res, d["name"], d["true_name"])
        rows.append((tag, m, res))
        if verbose:
            print(
                f"  {tag:22} P {m['presence']:.2f} L {m['localization']:.2f} "
                f"G {m['geometric']:.2f} I {m['identification']:.0f} total {m['total']:.3f}",
                flush=True,
            )
    return rows


def validate_caches(sample):
    """Fail before any identification if inputs are missing or inconsistent."""
    resolved = {}
    for _, record in sample.iterrows():
        scene = record["Id"]
        tag = f"val_{scene}"
        folder = resolve_cache(tag)
        if folder is None:
            raise RuntimeError(f"Missing cache for {tag}; refusing to start an expensive search. "
                               "Set Q5_CACHE to the completed cache directory.")
        with (folder / f"{tag}.pkl").open("rb") as handle:
            names, hyps = pickle.load(handle)
        maps = np.load(folder / f"{tag}_maps.npy", mmap_mode="r")
        expected = [f"patch_{i:02d}" for i in range(1, int(record["n_patches"]) + 1)]
        scene_dir = ROOT / "validation" / scene
        if names != expected or len(hyps) != len(expected):
            raise ValueError(f"{tag}: cache patch names/count do not match sample submission")
        with Image.open(scene_dir / f"{scene}_image.png") as im:
            w, h = im.size
        valid_shapes = {((h - 31 + pool - 1) // pool, (w - 31 + pool - 1) // pool)
                        for pool in (2, 4)}
        if maps.ndim != 3 or maps.shape[0] != len(names) or maps.shape[1:] not in valid_shapes:
            raise ValueError(f"{tag}: unexpected map shape {maps.shape}")
        if not np.issubdtype(maps.dtype, np.floating):
            raise ValueError(f"{tag}: expected floating-point maps")
        for name in names:
            if not (scene_dir / "patches" / f"{name}.png").is_file():
                raise FileNotFoundError(f"{tag}: missing {name}.png")
        resolved[tag] = folder
    return resolved


def smoke():
    """Validate every cache header and patch list; exercise eight claim planes."""
    import resource
    sample = pd.read_csv(ROOT / "sample_submission.csv")
    resolved = validate_caches(sample)
    paths = [folder / f"{tag}_maps.npy" for tag, folder in resolved.items()]
    path = min(paths, key=lambda p: p.stat().st_size)
    maps = np.load(path, mmap_mode="r")
    scene = path.name.removeprefix("val_").removesuffix("_maps.npy")
    with Image.open(ROOT / "validation" / scene / f"{scene}_image.png") as im:
        shape = (im.height, im.width)
    claims = Claims(maps[:min(8, len(maps))], PARAMS["claim_radius"], infer_pool(maps, shape))
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    rss_mb = rss / (1024 ** 2 if sys.platform == "darwin" else 1024)
    print(f"smoke ok: {len(resolved)} validated caches; {path.name} "
          f"maps {maps.shape}, claims {claims.dil.shape}, peak RSS {rss_mb:.0f} MiB", flush=True)


def _atomic_json(path, payload):
    partial = path.with_name(path.name + ".partial")
    partial.write_text(json.dumps(payload, sort_keys=True))
    os.replace(partial, path)


def _run_signature(patterns, p):
    digest = hashlib.sha256()
    digest.update(json.dumps(dict(params=p, fine=FINE), sort_keys=True).encode())
    for filename in ("q5_hpc.py", "q5_identify.py", "q5_search.py", "q5_synth.py", "q5_constellation.py"):
        digest.update((HERE / filename).read_bytes())
    for name, nodes in sorted(patterns.items()):
        digest.update(name.encode())
        digest.update(np.asarray(nodes).tobytes())
    return digest.hexdigest()


def _scene_signature(run_signature, folder, tag, scene_dir, record):
    # Cache map files are large: use file metadata rather than rereading GBs.
    paths = [folder / f"{tag}.pkl", folder / f"{tag}_maps.npy",
             scene_dir / f"{scene_dir.name}_image.png"]
    paths += sorted((scene_dir / "patches").glob("patch_*.png"))
    metadata = [(str(path.resolve()), path.stat().st_size, path.stat().st_mtime_ns) for path in paths]
    payload = [run_signature, str(record["Id"]), int(record["n_patches"]), metadata]
    return hashlib.sha256(json.dumps(payload).encode()).hexdigest()


def write_submission(patterns, out_path, p=PARAMS, only=None):
    sample = pd.read_csv(ROOT / "sample_submission.csv")
    if only is not None:
        sample = sample[sample["Id"].isin(only)]
    if sample.empty:
        raise ValueError("No requested scenes found")
    # Submission never calls search_scene: missing caches cannot start a search.
    resolved = validate_caches(sample)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = out_path.with_suffix(".checkpoints")
    checkpoint_dir.mkdir(exist_ok=True)
    run_signature = _run_signature(patterns, p)
    rows = []
    for _, record in sample.iterrows():
        scene = record["Id"]
        scene_dir = ROOT / "validation" / scene
        tag = f"val_{scene}"
        folder = resolved[tag]
        signature = _scene_signature(run_signature, folder, tag, scene_dir, record)
        checkpoint = checkpoint_dir / f"{scene}.json"
        if checkpoint.exists():
            saved = json.loads(checkpoint.read_text())
            if saved.get("signature") == signature:
                rows.append(saved["row"])
                print(f"{scene}: resumed completed prediction", flush=True)
                continue
        t = time.time()
        with (folder / f"{tag}.pkl").open("rb") as handle:
            names, hyps = pickle.load(handle)
        maps = np.load(folder / f"{tag}_maps.npy", mmap_mode="r")
        sky = load_sky(scene_dir)
        queries = load_queries(scene_dir, names)
        name, fig, table = figure_stage(hyps, maps, sky, patterns, p)
        if table:
            fig["margin"] = table[0][0] - (table[1][0] if len(table) > 1 else -10.0)
        res = finalize(names, hyps, fig, sky, queries, p)
        row = format_row(scene, int(record["n_patches"]), res, name)
        _atomic_json(checkpoint, dict(signature=signature, row=row))
        rows.append(row)
        n_fig = sum(1 for v in res.values() if v[2] == 1)
        top = [(round(r[0], 2), r[1]) for r in table[:3]]
        print(f"{scene}: {name:16} figure {n_fig} top {top} {time.time() - t:.0f}s; saved", flush=True)
        # Release the previous scene before constructing the next claim stack.
        del fig, table, maps, sky, queries
    partial = out_path.with_name(out_path.name + ".partial")
    pd.DataFrame(rows, columns=sample.columns).to_csv(partial, index=False)
    os.replace(partial, out_path)
    print(f"wrote complete {len(rows)}-scene submission: {out_path}", flush=True)


def sweep(patterns, scenes, base=None):
    base = dict(PARAMS if base is None else base)
    cache = {}
    print("building proposals (once)...", flush=True)
    prepare(scenes, patterns, base, verbose=False, proposal_cache=cache)
    configs = []
    for rule in ("excess", "blend"):
        for select in ("raw", "residual"):
            for high in (0.85, 0.88, 0.92):
                for spare in (True, False):
                    configs.append(dict(rule=rule, select=select, high=high, spare_fill=spare))
    results = []
    for cfg in configs:
        p = dict(base, **cfg)
        prep = prepare(scenes, patterns, p, verbose=False, proposal_cache=cache)
        rows = score_prepared(prep, p, verbose=False)
        real = [m["total"] for k, m, _ in rows if k.startswith("train_")]
        syn = [m["total"] for k, m, _ in rows if k.startswith("syn_")]
        real_id = [m["identification"] for k, m, _ in rows if k.startswith("train_")]
        syn_id = [m["identification"] for k, m, _ in rows if k.startswith("syn_")]
        score = (np.mean(real) if real else 0) * 3 + (np.mean(syn) if syn else 0) * 16
        # Do not keep `prep`. Each one holds dilated claim maps for every scene,
        # and 24 of those copies blew past 48 GB on job 18399338.
        results.append((cfg, score, real, syn, real_id, syn_id))
        del prep
        print(
            f"{cfg}  real {np.mean(real):.3f} ({sum(real_id)}/{len(real_id)} id)  "
            f"syn {np.mean(syn):.3f} ({sum(syn_id)}/{len(syn_id)} id)",
            flush=True,
        )
    best = max(results, key=lambda r: r[1])
    print("best", best[0], flush=True)
    (HERE / "cache_v3").mkdir(exist_ok=True)
    (HERE / "cache_v3" / "best_hpc.pkl").write_bytes(pickle.dumps(best[0]))
    prep = prepare(scenes, patterns, dict(base, **best[0]), verbose=False, proposal_cache=cache)
    return best[0], prep


def search_missing(patterns=None):
    todo = [(f"train_{d.name}", d) for d in sorted((ROOT / "train").iterdir()) if d.is_dir()]
    todo += [(f"val_{d.name}", d) for d in sorted((ROOT / "validation").iterdir()) if d.is_dir()]
    syn = HERE / "synthetic"
    if syn.exists():
        todo += [(f"syn_{d.name}", d) for d in sorted(syn.iterdir()) if d.is_dir()]
    for tag, folder in todo:
        fine_done = (CACHE / f"{tag}.pkl").exists() and (CACHE / f"{tag}_maps.npy").exists()
        if fine_done or (resolve_cache(tag) is not None and not FINE):
            print(f"cached {tag}", flush=True)
            continue
        print(f"searching {tag} workers={WORKERS} fine={int(FINE)}", flush=True)
        t = time.time()
        search_scene(folder, tag, force=True)
        print(f"done {tag} {time.time() - t:.0f}s", flush=True)


if __name__ == "__main__":
    args = set(sys.argv[1:])
    patterns = load_patterns(ROOT / "patterns")
    if "--search-all" in args or "--all" in args:
        search_missing(patterns)
    if "--sweep" in args or "--all" in args:
        cfg, prep = sweep(patterns, labelled_scenes())
        PARAMS.update(cfg)
        rows = score_prepared(prep, dict(PARAMS), verbose=True)
        print("mean", np.mean([m["total"] for _, m, _ in rows]), flush=True)
    elif "--eval" in args:
        prep = prepare(labelled_scenes(), patterns)
        rows = score_prepared(prep, verbose=True)
        print("mean", np.mean([m["total"] for _, m, _ in rows]), flush=True)
    if "--smoke" in args or "--submit" in args or "--all" in args:
        print("using", PARAMS, flush=True)
        if "--smoke" in args and "--submit" not in args and "--all" not in args:
            smoke()
        else:
            write_submission(patterns, HERE / "submission_hpc.csv")
    if not args:
        print("usage: q5_hpc.py [--search-all] [--eval] [--sweep] [--submit] [--all]", flush=True)
        print(f"Q5_FINE={int(FINE)} Q5_WORKERS={WORKERS} CACHE={CACHE}", flush=True)
