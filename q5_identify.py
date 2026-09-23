"""Stage 2: register the 48 pattern drawings to the candidate star locations.

A similarity transform (with optional reflection) maps a drawing onto the sky.
Its evidence has two parts:

* patch inliers: pattern nodes that land on a candidate location, at most one
  per patch;
* sky support: every node of the real figure is a bright star in the sky image,
  including nodes that were never issued as patches.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image

CELL = 0.06
MIN_BASE_PX = 60.0


def load_pattern_nodes(path):
    image = np.array(Image.open(path).convert("RGBA"))
    rgb = image[:, :, :3]
    white = (rgb[:, :, 0] > 200) & (rgb[:, :, 1] > 200) & (rgb[:, :, 2] > 200)
    count, _, stats, centroids = cv2.connectedComponentsWithStats(white.astype(np.uint8), 8)
    nodes = [centroids[i] for i in range(1, count) if stats[i, cv2.CC_STAT_AREA] >= 4]
    return np.asarray(nodes, np.float32).reshape(-1, 2)


def load_patterns(patterns_dir):
    patterns = {}
    for path in sorted(Path(patterns_dir).glob("*_pattern.png")):
        nodes = load_pattern_nodes(path)
        if len(nodes) >= 2:
            patterns[path.stem.replace("_pattern", "")] = nodes
    return patterns


class SkySupport:
    """Percentile of 'brightest star within ~10 px' at any sky location."""

    def __init__(self, sky, radius=10):
        g = sky.astype(np.float32)
        dog = cv2.GaussianBlur(g, (0, 0), 1.5) - cv2.GaussianBlur(g, (0, 0), 6.0)
        disk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
        self.near = cv2.dilate(dog, disk)
        self.ref = np.sort(self.near[::7, ::7].ravel())
        self.h, self.w = sky.shape

    def percentile(self, points):
        points = np.asarray(points, np.float32).reshape(-1, 2)
        inside = (points[:, 0] >= 0) & (points[:, 0] < self.w) & (points[:, 1] >= 0) & (points[:, 1] < self.h)
        out = np.zeros(len(points), np.float32)
        if inside.any():
            xi = np.clip(np.round(points[inside, 0]).astype(int), 0, self.w - 1)
            yi = np.clip(np.round(points[inside, 1]).astype(int), 0, self.h - 1)
            out[inside] = np.searchsorted(self.ref, self.near[yi, xi]) / len(self.ref)
        return out, inside


def _similarity_from_pairs(p0, p1, q0, q1):
    """Map p0->q0, p1->q1. Returns (scale*R, t)."""
    vp = p1 - p0
    vq = q1 - q0
    lp = np.hypot(*vp) + 1e-9
    lq = np.hypot(*vq)
    scale = lq / lp
    angle = np.arctan2(vq[1], vq[0]) - np.arctan2(vp[1], vp[0])
    c, s = np.cos(angle), np.sin(angle)
    A = scale * np.array([[c, -s], [s, c]], np.float64)
    t = q0 - A @ p0
    return A, t


def _fit_similarity_lsq(src, dst):
    """Least-squares similarity (Umeyama, no reflection; reflection handled outside)."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    s0, d0 = src - mu_s, dst - mu_d
    var = (s0 ** 2).sum() / len(src)
    cov = d0.T @ s0 / len(src)
    U, S, Vt = np.linalg.svd(cov)
    D = np.eye(2)
    if np.linalg.det(U @ Vt) < 0:
        D[1, 1] = -1
    R = U @ D @ Vt
    scale = np.trace(np.diag(S) @ D) / max(var, 1e-9)
    A = scale * R
    t = mu_d - A @ mu_s
    return A, t


def _match(pred, points, owners, tol):
    """Greedy one-to-one node/patch matching. Returns list of (node, point)."""
    d = np.hypot(points[:, None, 0] - pred[None, :, 0], points[:, None, 1] - pred[None, :, 1])
    cand = np.argwhere(d <= tol)
    if len(cand) == 0:
        return []
    order = np.argsort(d[cand[:, 0], cand[:, 1]])
    used_owner, used_node, pairs = set(), set(), []
    for k in order:
        pi, ni = int(cand[k, 0]), int(cand[k, 1])
        o = int(owners[pi])
        if o in used_owner or ni in used_node:
            continue
        used_owner.add(o)
        used_node.add(ni)
        pairs.append((ni, pi))
    return pairs


class Registrar:
    def __init__(self, points, owners, weights, support, tol=20.0, lam=1.5):
        self.points = np.asarray(points, np.float64).reshape(-1, 2)
        self.owners = np.asarray(owners)
        self.weights = np.asarray(weights, np.float64)
        self.support = support
        self.tol = tol
        self.lam = lam
        self._build_table()

    def _build_table(self):
        P = self.points
        n = len(P)
        a_idx, b_idx = np.nonzero(np.ones((n, n), bool))
        keep = (self.owners[a_idx] != self.owners[b_idx])
        a_idx, b_idx = a_idx[keep], b_idx[keep]
        vec = P[b_idx] - P[a_idx]
        length = np.hypot(vec[:, 0], vec[:, 1])
        keep = length >= MIN_BASE_PX
        a_idx, b_idx, vec, length = a_idx[keep], b_idx[keep], vec[keep], length[keep]
        self.base_a, self.base_b = a_idx, b_idx
        if len(a_idx) == 0:
            self.keys = np.zeros(0, np.int64)
            self.key_base = np.zeros(0, np.int64)
            return
        c = vec[:, 0] / length
        s = vec[:, 1] / length
        rel = P[None, :, :] - P[a_idx][:, None, :]
        u = (c[:, None] * rel[:, :, 0] + s[:, None] * rel[:, :, 1]) / length[:, None]
        v = (-s[:, None] * rel[:, :, 0] + c[:, None] * rel[:, :, 1]) / length[:, None]
        valid = (self.owners[None, :] != self.owners[a_idx][:, None]) & (
            self.owners[None, :] != self.owners[b_idx][:, None]
        )
        valid &= (np.abs(u) < 6) & (np.abs(v) < 6)
        base_ids = np.broadcast_to(np.arange(len(a_idx))[:, None], u.shape)[valid]
        qu = np.floor(u[valid] / CELL).astype(np.int64)
        qv = np.floor(v[valid] / CELL).astype(np.int64)
        keys = (qu + 1000) * 4096 + (qv + 1000)
        order = np.argsort(keys, kind="stable")
        self.keys = keys[order]
        self.key_base = base_ids[order]

    def _votes(self, nodes, i, j):
        vec = nodes[j] - nodes[i]
        length = np.hypot(*vec)
        c, s = vec / length
        rel = nodes - nodes[i]
        u = (c * rel[:, 0] + s * rel[:, 1]) / length
        v = (-s * rel[:, 0] + c * rel[:, 1]) / length
        mask = np.ones(len(nodes), bool)
        mask[[i, j]] = False
        qu = np.floor(u[mask] / CELL).astype(np.int64)
        qv = np.floor(v[mask] / CELL).astype(np.int64)
        lookups = []
        for du in (-1, 0, 1):
            for dv in (-1, 0, 1):
                lookups.append((qu + du + 1000) * 4096 + (qv + dv + 1000))
        lookups = np.concatenate(lookups)
        lo = np.searchsorted(self.keys, lookups, "left")
        hi = np.searchsorted(self.keys, lookups, "right")
        counts = hi - lo
        if counts.sum() == 0:
            return None
        starts = np.repeat(lo, counts)
        offsets = np.arange(counts.sum()) - np.repeat(np.cumsum(counts) - counts, counts)
        hits = self.key_base[starts + offsets]
        return np.bincount(hits, minlength=len(self.base_a))

    def evaluate(self, nodes, A, t):
        """Transform quality from patch inliers only; sky support is reported, not optimised."""
        pred = nodes @ A.T + t
        pairs = _match(pred, self.points, self.owners, self.tol)
        if len(pairs) >= 3:
            src = nodes[[p[0] for p in pairs]]
            dst = self.points[[p[1] for p in pairs]]
            A2, t2 = _fit_similarity_lsq(src, dst)
            pred2 = nodes @ A2.T + t2
            pairs2 = _match(pred2, self.points, self.owners, self.tol)
            if len(pairs2) >= len(pairs):
                A, t, pred, pairs = A2, t2, pred2, pairs2
        inliers = float(sum(self.weights[p[1]] for p in pairs))
        if pairs:
            gaps = [np.hypot(*(pred[n] - self.points[p])) for n, p in pairs]
            error = float(np.mean(gaps))
        else:
            error = self.tol
        extent = float(np.ptp(pred[:, 0]) + np.ptp(pred[:, 1]))
        fit_score = inliers - 0.01 * error
        if extent < 400:
            fit_score -= 5.0
        return fit_score, pairs, pred, inliers, error

    def sky_support(self, pred, pairs, bright=0.8, chance=0.2):
        """Unissued nodes that land on bright stars, above the rate expected by chance."""
        pct, inside = self.support.percentile(pred)
        matched = np.zeros(len(pred), bool)
        matched[[p[0] for p in pairs]] = True
        hits = (pct > bright) & inside
        return float(np.sum(hits[~matched]) - chance * np.sum(~matched))

    def best_fit(self, nodes, top_bases=4, max_pairs=160, rng_seed=0, keep=8):
        """Top transforms by patch inliers, each re-ranked by sky support."""
        if len(self.keys) == 0 or len(nodes) < 2:
            return None
        found = []
        n = len(nodes)
        ii, jj = np.triu_indices(n, 1)
        span = max(np.ptp(nodes[:, 0]), np.ptp(nodes[:, 1]), 1.0)
        sep = np.hypot(*(nodes[jj] - nodes[ii]).T)
        good = np.flatnonzero(sep >= 0.12 * span)
        if len(good) > max_pairs:
            good = np.random.default_rng(rng_seed).choice(good, max_pairs, replace=False)
        for reflect in (False, True):
            src = nodes.astype(np.float64).copy()
            if reflect:
                src[:, 0] = -src[:, 0]
            for g in good:
                i, j = int(ii[g]), int(jj[g])
                for a_node, b_node in ((i, j), (j, i)):
                    votes = self._votes(src, a_node, b_node) if n > 2 else None
                    if votes is None:
                        if n > 2:
                            continue
                        candidates = np.arange(len(self.base_a))[:200]
                    else:
                        candidates = np.argsort(-votes)[:top_bases]
                        candidates = candidates[votes[candidates] > 0]
                    for base in candidates:
                        A, t = _similarity_from_pairs(
                            src[a_node], src[b_node],
                            self.points[self.base_a[base]], self.points[self.base_b[base]],
                        )
                        found.append(self.evaluate(src, A, t) + (reflect,))
        if not found:
            return None
        found.sort(key=lambda r: -r[0])
        top_inliers = found[0][3]
        best = None
        seen = set()
        for fit_score, pairs, pred, inliers, error, reflect in found:
            if inliers < top_inliers - 1.0 or len(seen) >= keep:
                break
            key = tuple(sorted(pairs))
            if key in seen:
                continue
            seen.add(key)
            support = self.sky_support(pred, pairs)
            total = inliers + self.lam * support
            if best is None or total > best[0]:
                best = (total, pairs, pred, inliers, support, reflect)
        return best

    def top_preds(self, nodes, k=6, **kwargs):
        """Up to k distinct predicted placements, best first."""
        if len(self.keys) == 0 or len(nodes) < 2:
            return []
        found = []
        n = len(nodes)
        ii, jj = np.triu_indices(n, 1)
        span = max(np.ptp(nodes[:, 0]), np.ptp(nodes[:, 1]), 1.0)
        sep = np.hypot(*(nodes[jj] - nodes[ii]).T)
        good = np.flatnonzero(sep >= 0.12 * span)
        max_pairs = kwargs.get("max_pairs", 160)
        rng_seed = kwargs.get("rng_seed", 0)
        top_bases = kwargs.get("top_bases", 4)
        if len(good) > max_pairs:
            good = np.random.default_rng(rng_seed).choice(good, max_pairs, replace=False)
        for reflect in (False, True):
            src = nodes.astype(np.float64).copy()
            if reflect:
                src[:, 0] = -src[:, 0]
            for g in good:
                i, j = int(ii[g]), int(jj[g])
                for a_node, b_node in ((i, j), (j, i)):
                    votes = self._votes(src, a_node, b_node) if n > 2 else None
                    if votes is None:
                        if n > 2:
                            continue
                        candidates = np.arange(len(self.base_a))[:200]
                    else:
                        candidates = np.argsort(-votes)[:top_bases]
                        candidates = candidates[votes[candidates] > 0]
                    for base in candidates:
                        A, t = _similarity_from_pairs(
                            src[a_node], src[b_node],
                            self.points[self.base_a[base]], self.points[self.base_b[base]],
                        )
                        found.append(self.evaluate(src, A, t))
        if not found:
            return []
        found.sort(key=lambda r: -r[0])
        out, seen = [], set()
        for fit_score, pairs, pred, inliers, error in found:
            key = tuple(np.round(pred.mean(0) / 25.0).astype(int))
            if key in seen:
                continue
            seen.add(key)
            out.append(pred)
            if len(out) >= k:
                break
        return out


def identify(patterns, points, owners, weights, support, tol=20.0, lam=1.5, min_inliers=3):
    """Return (name, pairs, pred, score, table of all pattern scores)."""
    if len(points) < 3:
        return "unknown", [], None, 0.0, []
    reg = Registrar(points, owners, weights, support, tol=tol, lam=lam)
    table = []
    for name, nodes in patterns.items():
        if len(nodes) < 3:
            continue
        fit = reg.best_fit(nodes)
        if fit is None:
            continue
        score, pairs, pred, inliers, sup, reflect = fit
        table.append((score, name, pairs, pred, inliers, sup))
    if not table:
        return "unknown", [], None, 0.0, []
    table.sort(key=lambda r: -r[0])
    score, name, pairs, pred, inliers, sup = table[0]
    if len(pairs) < min_inliers:
        return "unknown", [], None, score, table
    return name, pairs, pred, score, table
