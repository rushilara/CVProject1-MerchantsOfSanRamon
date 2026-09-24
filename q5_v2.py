"""Second attempt: keep near-tied locations per patch, let the constellation choose.

Stage 1 (q5_search): rotation/scale sweep of normalised cross-correlation.
Stage 2 (q5_identify): register each pattern drawing to the candidate cloud,
at most one location per patch, inliers weighted by star brightness.
Stage 3 (here): report figure stars at their registered location (m = 1),
other present patches at their best peak (m = 0), and decide present/absent.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from scipy.optimize import linear_sum_assignment

import q5_search
from q5_constellation import format_row, parse_truth, score_scene
from q5_identify import Registrar, SkySupport, load_patterns

ROOT = q5_search.ROOT

PARAMS = dict(
    present_ncc=0.90,     # present if best peak is at least this strong ...
    moderate_ncc=0.55,    # ... or at least this strong and clear of rivals
    moderate_margin=0.04,
    tie_gap=0.03,         # locations this close to the best peak are real alternatives
    max_alternatives=8,
    min_ncc_for_fit=0.55,
    bright_pct=0.8,       # figure stars sit on bright sky
    tol=20.0,
    top_patterns=5,
    node_window=18,       # half-width of the local search box around a node
    node_ncc=0.70,        # a patch must correlate this well to claim a node
)


def choose_present(hyps, p):
    if not hyps:
        return False
    best = hyps[0]
    rival = next((h for h in hyps[1:] if np.hypot(h[1] - best[1], h[2] - best[2]) > 22), None)
    margin = best[0] - (rival[0] if rival else 0.0)
    return best[0] >= p["present_ncc"] or (best[0] >= p["moderate_ncc"] and margin >= p["moderate_margin"])


def warp_bank(query):
    query = query.astype(np.float32)
    return [q5_search.rotate_scale(query, a, s) for s in q5_search.SCALES for a in q5_search.ANGLES]


def node_assignment(sky_f, banks, pred, candidates, p):
    """Local rotation sweep of each candidate patch around each predicted node.

    Returns {patch_index: (ncc, x, y)} from a one-to-one assignment.
    """
    h, w = sky_f.shape
    half = p["node_window"] + 16
    scores = np.full((len(pred), len(candidates)), -1.0)
    where = {}
    for ni, (x, y) in enumerate(pred):
        xi, yi = int(round(x)), int(round(y))
        if xi - half < 0 or yi - half < 0 or xi + half > w or yi + half > h:
            continue
        window = sky_f[yi - half:yi + half, xi - half:xi + half]
        for ci, index in enumerate(candidates):
            best = -1.0
            best_xy = None
            for warped in banks[index]:
                res = cv2.matchTemplate(window, warped, cv2.TM_CCOEFF_NORMED)
                j = int(np.argmax(res))
                v = float(res.flat[j])
                if v > best:
                    py, px = divmod(j, res.shape[1])
                    best = v
                    best_xy = (xi - half + px + 16, yi - half + py + 16)
            scores[ni, ci] = best
            where[(ni, ci)] = best_xy
    cost = np.where(scores >= p["node_ncc"], -scores, 10.0)
    rows, cols = linear_sum_assignment(cost)
    out = {}
    for r, c in zip(rows, cols):
        if scores[r, c] >= p["node_ncc"]:
            x, y = where[(r, c)]
            out[candidates[c]] = (float(scores[r, c]), float(x), float(y))
    return out


def candidate_cloud(hyps, support, p):
    points, owners, backing = [], [], []
    for index, hs in enumerate(hyps):
        if not hs or hs[0][0] < p["min_ncc_for_fit"]:
            continue
        best = hs[0][0]
        for h in hs[: p["max_alternatives"]]:
            if h[0] < best - p["tie_gap"]:
                continue
            points.append((h[1], h[2]))
            owners.append(index)
            backing.append(h)
    points = np.asarray(points, np.float64).reshape(-1, 2)
    owners = np.asarray(owners)
    backing = list(backing)
    if len(points):
        pct, _ = support.percentile(points)
        keep = pct >= p["bright_pct"]
        points, owners = points[keep], owners[keep]
        backing = [b for b, k in zip(backing, keep) if k]
    return points, owners, backing


def scene_prediction(names, hyps, sky, patterns, queries=None, p=PARAMS, return_debug=False):
    constellation, figure_loc, table = figure_stage(hyps, sky, patterns, queries, p)
    results = finalize(names, hyps, figure_loc, p)
    if return_debug:
        return results, constellation, table
    return results, constellation


def finalize(names, hyps, figure_loc, p=PARAMS):
    results = {}
    for index, name in enumerate(names):
        if index in figure_loc:
            h = figure_loc[index]
            results[name] = (int(round(h[1])), int(round(h[2])), 1)
        elif choose_present(hyps[index], p):
            h = hyps[index][0]
            results[name] = (int(round(h[1])), int(round(h[2])), 0)
        else:
            results[name] = (-1, -1, -1)
    return results


def figure_stage(hyps, sky, patterns, queries=None, p=PARAMS):
    support = SkySupport(sky)
    sky_f = sky.astype(np.float32)
    points, owners, backing = candidate_cloud(hyps, support, p)

    table = []
    if len(points) >= 3:
        reg = Registrar(points, owners, np.ones(len(points)), support, tol=p["tol"], lam=0.0)
        for name, nodes in patterns.items():
            if len(nodes) < 3:
                continue
            fit = reg.best_fit(nodes, top_bases=10, max_pairs=300)
            if fit is not None:
                table.append((fit[3], name, fit[1], fit[2]))
        table.sort(key=lambda r: -r[0])

    constellation = "unknown"
    figure_loc = {}
    if table and len(table[0][2]) >= 3:
        # Verify the leading drawings by claiming their nodes with a local search.
        candidates = sorted(set(int(o) for o in owners))
        banks = {i: warp_bank(queries[i]) for i in candidates} if queries is not None else None
        best = None
        for inliers, name, pairs, pred in table[: p["top_patterns"]]:
            if banks is None:
                claimed = {int(owners[q]): (backing[q][0], backing[q][1], backing[q][2]) for _, q in pairs}
            else:
                claimed = node_assignment(sky_f, banks, pred, candidates, p)
            # Per node, so large drawings do not win just by offering more bright targets.
            evidence = sum(v[0] for v in claimed.values()) / len(pred) + 0.02 * inliers
            if best is None or evidence > best[0]:
                best = (evidence, name, claimed)
        _, constellation, figure_loc = best
    return constellation, figure_loc, table


def load_sky(scene_dir):
    scene_dir = Path(scene_dir)
    return np.array(Image.open(scene_dir / f"{scene_dir.name}_image.png").convert("L"))


def load_queries(scene_dir, names):
    scene_dir = Path(scene_dir)
    return [np.array(Image.open(scene_dir / "patches" / f"{n}.png").convert("L")) for n in names]


def evaluate_training(patterns, p=PARAMS, verbose=True):
    gt = pd.read_csv(ROOT / "train_ground_truth.csv")
    totals = []
    for _, record in gt.iterrows():
        scene = record["Id"]
        scene_dir = ROOT / "train" / scene
        names, hyps = q5_search.search_scene(scene_dir, tag=f"train_{scene}")
        sky = load_sky(scene_dir)
        queries = load_queries(scene_dir, names)
        results, constellation, table = scene_prediction(
            names, hyps, sky, patterns, queries, p, return_debug=True
        )
        truth = {f"patch_{i:02d}": parse_truth(record[f"patch_{i:02d}"]) for i in range(1, int(record["n_patches"]) + 1)}
        metrics = score_scene(truth, results, constellation, record["constellation"])
        totals.append(metrics["total"])
        if verbose:
            fig_ok = sum(
                1 for n, t in truth.items()
                if t and t[2] == 1 and results[n][2] == 1 and np.hypot(results[n][0] - t[0], results[n][1] - t[1]) <= 12
            )
            n_fig = sum(1 for t in truth.values() if t and t[2] == 1)
            top = [(round(r[0], 1), r[1], len(r[2])) for r in table[:3]]
            print(
                f"{scene:9} -> {constellation:16} P {metrics['presence']:.2f} L {metrics['localization']:.2f} "
                f"G {metrics['geometric']:.2f} I {metrics['identification']:.0f} total {metrics['total']:.3f} "
                f"| figure stars placed {fig_ok}/{n_fig} | top {top}",
                flush=True,
            )
    mean = float(np.mean(totals))
    if verbose:
        print(f"mean training score {mean:.3f}", flush=True)
    return mean


def write_submission(patterns, out_path, p=PARAMS):
    sample = pd.read_csv(ROOT / "sample_submission.csv")
    rows = []
    for _, record in sample.iterrows():
        scene = record["Id"]
        scene_dir = ROOT / "validation" / scene
        names, hyps = q5_search.search_scene(scene_dir, tag=f"val_{scene}")
        sky = load_sky(scene_dir)
        queries = load_queries(scene_dir, names)
        results, constellation = scene_prediction(names, hyps, sky, patterns, queries, p)
        rows.append(format_row(scene, int(record["n_patches"]), results, constellation))
        n_present = sum(1 for v in results.values() if v[2] != -1)
        n_fig = sum(1 for v in results.values() if v[2] == 1)
        print(f"{scene}: {constellation:16} present {n_present}/{len(names)} figure {n_fig}", flush=True)
    pd.DataFrame(rows, columns=sample.columns).to_csv(out_path, index=False)
    print(f"wrote {out_path}", flush=True)


if __name__ == "__main__":
    patterns = load_patterns(ROOT / "patterns")
    evaluate_training(patterns)
    if "--submit" in sys.argv:
        write_submission(patterns, Path(__file__).resolve().parent / "submission_v2.csv")
