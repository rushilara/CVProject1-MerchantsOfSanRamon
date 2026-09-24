"""Third attempt: verify every drawing with per-patch claim maps.

claim(p, x, y) is the best correlation of patch p over all rotations and
scales at sky position (x, y), from a 4x-pooled map saved in stage 1.

For each drawing, similarity transforms are proposed from pairs of candidate
locations (two nodes placed exactly on two patches' candidates), scored
quickly with the best claim over all patches, then the best few are scored
with a one-to-one node/patch assignment.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from scipy.ndimage import maximum_filter
from scipy.optimize import linear_sum_assignment

import q5_search
import q5_search3
from q5_constellation import format_row, parse_truth, score_scene
from q5_identify import SkySupport, load_patterns

HERE = Path(__file__).resolve().parent
ROOT = HERE / "participant"
POOL = q5_search3.POOL

PARAMS = dict(
    tie_gap=0.03,
    max_alternatives=8,
    min_ncc_for_fit=0.55,
    bright_pct=0.8,
    min_base_px=60.0,
    max_node_pairs=30,
    keep_transforms=40,
    claim_radius=2,        # pooled cells (x4 px) around a node
    claim_min=0.70,
    quick_level=0.85,
    rule="excess",
    select="raw",
    high=0.90,
    present_ncc=0.90,
    moderate_ncc=0.55,
    moderate_margin=0.04,
    spare_fill=False,
)


class Claims:
    def __init__(self, maps, radius):
        self.dil = np.stack([
            maximum_filter(np.asarray(m, dtype=np.float16), size=2 * radius + 1, mode="nearest")
            for m in maps
        ])
        self.best = self.dil.max(axis=0).astype(np.float32)
        self.shape = self.best.shape

    def cells(self, xy):
        xy = np.asarray(xy, np.float64)
        j = np.round((xy[..., 0] - 17.0) / POOL).astype(np.int64)
        i = np.round((xy[..., 1] - 17.0) / POOL).astype(np.int64)
        inside = (i >= 0) & (i < self.shape[0]) & (j >= 0) & (j < self.shape[1])
        return np.clip(i, 0, self.shape[0] - 1), np.clip(j, 0, self.shape[1] - 1), inside

    def best_at(self, xy):
        i, j, inside = self.cells(xy)
        v = self.best[i, j]
        return np.where(inside, v, -1.0)

    def matrix(self, xy):
        """(nodes, patches) claim values."""
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


def bases(pts, owners, min_len):
    n = len(pts)
    a, b = np.nonzero(np.ones((n, n), bool))
    keep = owners[a] != owners[b]
    a, b = a[keep], b[keep]
    length = np.hypot(*(pts[b] - pts[a]).T)
    keep = length >= min_len
    return a[keep], b[keep]


def propose(nodes, pts, base_a, base_b, claims, p, rng):
    """Vectorised similarity proposals scored by the best claim at every node."""
    n = len(nodes)
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
            pred = np.stack([px, py], axis=-1)
            extent = np.ptp(px, axis=1) + np.ptp(py, axis=1)
            quick = (claims.best_at(pred) >= p["quick_level"]).sum(axis=1).astype(np.float64)
            quick[extent < 400] = -1
            top = np.argsort(-quick)[: p["keep_transforms"]]
            for t in top:
                proposals.append((quick[t], pred[t]))
    proposals.sort(key=lambda r: -r[0])
    return proposals[: p["keep_transforms"]]


def assign(pred, claims, p):
    m = claims.matrix(pred)
    cost = np.where(m >= p["claim_min"], -m, 10.0)
    rows, cols = linear_sum_assignment(cost)
    out = [(int(r), int(c), float(m[r, c])) for r, c in zip(rows, cols) if m[r, c] >= p["claim_min"]]
    return out


def evidence(pairs, n_nodes, p, null_high):
    vals = np.array([v for _, _, v in pairs]) if pairs else np.zeros(0)
    if p["rule"] == "mean":
        return vals.sum() / n_nodes
    if p["rule"] == "excess":
        return float((vals >= p["high"]).sum() - null_high * n_nodes)
    if p["rule"] == "blend":
        return float((vals >= p["high"]).sum() - null_high * n_nodes) + vals.sum() / n_nodes
    raise ValueError(p["rule"])


def null_high_rate(claims, support, p, rng, samples=3000):
    """How often a random position has a patch claiming it at the high level."""
    ys = rng.uniform(40, support.h - 40, samples)
    xs = rng.uniform(40, support.w - 40, samples)
    return float(np.mean(claims.best_at(np.column_stack([xs, ys])) >= p["high"]))


def scene_proposals(hyps, maps, sky, patterns, p=PARAMS, seed=0):
    rng = np.random.default_rng(seed)
    support = SkySupport(sky)
    claims = Claims(maps, p["claim_radius"])
    pts, owners = candidate_cloud(hyps, support, p)
    proposals = {}
    if len(pts) >= 2:
        base_a, base_b = bases(pts, owners, p["min_base_px"])
        if len(base_a):
            for name, nodes in patterns.items():
                if len(nodes) >= 3:
                    proposals[name] = [pred for _, pred in propose(nodes, pts, base_a, base_b, claims, p, rng)]
    return dict(support=support, claims=claims, proposals=proposals)


def figure_stage(hyps, maps, sky, patterns, p=PARAMS, seed=0, cached=None):
    """cached may hold only 'proposals'; claim maps are rebuilt so they are never kept for many scenes."""
    if cached is None:
        cached = scene_proposals(hyps, maps, sky, patterns, p, seed)
    proposals = cached["proposals"]
    claims = cached.get("claims") or Claims(maps, p["claim_radius"])
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
        # Most drawings are wrong, so their scores trace what chance gives a drawing of each size.
        from scipy.stats import theilslopes
        sizes = np.array([len(patterns[r[1]]) for r in table], float)
        scores = np.array([r[0] for r in table], float)
        slope, intercept, _, _ = theilslopes(scores, sizes)
        table = [(s - (intercept + slope * n),) + r[1:] for s, n, r in zip(scores, sizes, table)]
    table.sort(key=lambda r: -r[0])
    score, name, pred, pairs = table[0]
    figure = {}
    for node, patch, value in pairs:
        figure[patch] = (node, value)
    return name, {"pred": pred, "assigned": figure}, table


def refine_location(sky_f, query, x, y, half=14):
    """Exact local rotation sweep around (x, y)."""
    h, w = sky_f.shape
    xi, yi = int(round(x)), int(round(y))
    x0, y0 = max(0, xi - half - 16), max(0, yi - half - 16)
    x1, y1 = min(w, xi + half + 16), min(h, yi + half + 16)
    window = sky_f[y0:y1, x0:x1]
    if window.shape[0] < 33 or window.shape[1] < 33:
        return x, y, -1.0
    q = query.astype(np.float32)
    best = (-1.0, x, y)
    for s in q5_search.SCALES:
        for a in q5_search.ANGLES:
            res = cv2.matchTemplate(window, q5_search.rotate_scale(q, a, s), cv2.TM_CCOEFF_NORMED)
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
    if p["spare_fill"] and pred is not None:
        used_nodes = {node for node, _ in assigned.values()}
        spares = [i for i, n in enumerate(names) if results[n][2] == -1]
        for node in range(len(pred)):
            if node in used_nodes or not spares:
                continue
            x, y = pred[node]
            if 16 <= x < sky.shape[1] - 16 and 16 <= y < sky.shape[0] - 16:
                results[names[spares.pop()]] = (int(round(x)), int(round(y)), 1)
    return results


# ---------------------------------------------------------------------------
# scene loading
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
        for tag, folder in q5_search3.synthetic_scene_dirs():
            info = json.loads((folder / "truth.json").read_text())
            truth = {k: (tuple(v) if v else None) for k, v in info["patches"].items()}
            scenes.append((tag, folder, truth, info["constellation"]))
    return scenes


def prepare(scenes, patterns, p=PARAMS, verbose=True, proposal_cache=None):
    """Figure stage per scene. proposal_cache (dict) lets sweeps reuse proposals."""
    out = {}
    for tag, folder, truth, true_name in scenes:
        hyp_path = q5_search3.CACHE / f"{tag}.pkl"
        map_path = q5_search3.CACHE / f"{tag}_maps.npy"
        if not (hyp_path.exists() and map_path.exists()):
            continue
        names, hyps, maps = q5_search3.search_scene(folder, tag)
        sky = load_sky(folder)
        cached = None
        if proposal_cache is not None:
            if tag not in proposal_cache:
                proposal_cache[tag] = {"proposals": scene_proposals(hyps, maps, sky, patterns, p)["proposals"]}
            cached = proposal_cache[tag]
        name, fig, table = figure_stage(hyps, maps, sky, patterns, p, cached=cached)
        out[tag] = dict(names=names, hyps=hyps, sky=sky, folder=folder, truth=truth,
                        true_name=true_name, name=name, fig=fig, table=table)
        if verbose:
            rank = next((k for k, r in enumerate(table) if r[1] == true_name), None)
            top = [(round(r[0], 2), r[1]) for r in table[:3]]
            print(f"{tag:22} true {true_name:16} pred {name:16} true-rank {rank} top {top}", flush=True)
    return out


def score_prepared(prepared, p=PARAMS, verbose=False):
    rows = []
    for tag, d in prepared.items():
        queries = load_queries(d["folder"], d["names"])
        res = finalize(d["names"], d["hyps"], d["fig"], d["sky"], queries, p)
        m = score_scene(d["truth"], res, d["name"], d["true_name"])
        rows.append((tag, m))
        if verbose:
            print(f"  {tag:22} P {m['presence']:.2f} L {m['localization']:.2f} G {m['geometric']:.2f} "
                  f"I {m['identification']:.0f} total {m['total']:.3f}", flush=True)
    return rows


def write_submission(patterns, out_path, p=PARAMS):
    sample = pd.read_csv(ROOT / "sample_submission.csv")
    rows = []
    for _, record in sample.iterrows():
        scene = record["Id"]
        folder = ROOT / "validation" / scene
        names, hyps, maps = q5_search3.search_scene(folder, f"val_{scene}")
        sky = load_sky(folder)
        queries = load_queries(folder, names)
        name, fig, table = figure_stage(hyps, maps, sky, patterns, p)
        res = finalize(names, hyps, fig, sky, queries, p)
        rows.append(format_row(scene, int(record["n_patches"]), res, name))
        n_fig = sum(1 for v in res.values() if v[2] == 1)
        top = [(round(r[0], 2), r[1]) for r in table[:3]]
        print(f"{scene}: {name:16} figure {n_fig} top {top}", flush=True)
    pd.DataFrame(rows, columns=sample.columns).to_csv(out_path, index=False)
    print(f"wrote {out_path}", flush=True)


if __name__ == "__main__":
    patterns = load_patterns(ROOT / "patterns")
    if "--submit" in sys.argv:
        write_submission(patterns, HERE / "submission_v3.csv")
    else:
        prep = prepare(labelled_scenes(), patterns)
        rows = score_prepared(prep, verbose=True)
        print("mean", np.mean([m["total"] for _, m in rows]))
