"""Constellation detection pipeline.

Stage 1. Place each query patch, or reject it.
    The patch centre is the star to report, but the patch has been rotated,
    rescaled, blurred, and noised. Exhaustive normalised correlation over a
    grid of rotations and scales finds the sky locations that still look like
    the patch. A margin between the best peak and the next distinct peak
    decides present versus absent.

Stage 2. Name the figure.
    The 48 pattern drawings are point sets. A similarity transform that may
    include a reflection is fit to the recovered points. The drawing with the
    most uniquely matched stars is the constellation. Matched points are the
    figure (m = 1); other present points are field stars (m = 0).
"""

from __future__ import annotations

import ast
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

ROOT = Path(__file__).resolve().parent / "participant"

# 10 degrees is enough to keep the true alignment inside the correlation peak.
# Three scales cover the moderate resize applied to the patches.
ANGLES = tuple(range(0, 360, 10))
SCALES = (0.88, 1.0, 1.14)

# A second sky location this close is the same star, not a rival hypothesis.
MERGE_PX = 22.0

# A strong peak is kept even when another sky location is close, because
# dropping a real star costs localisation and figure recovery. A moderate
# peak is kept only when it stands clear of the next location.
NCC_PRESENT = 0.58
NCC_MARGIN = 0.08
NCC_FORCE = 0.72

# Pattern nodes must land this close to a recovered star.
PATTERN_TOL = 22.0
MIN_INLIERS = 5


def parse_truth(value):
    text = str(value).strip()
    if text in {"-1", "-1.0", "nan"}:
        return None
    x, y, m = ast.literal_eval(text)
    return float(x), float(y), int(m)


def load_gray(path):
    return np.array(Image.open(path).convert("L"))


def rotate_scale(query, angle, scale):
    """Rotate and scale about the patch centre. The centre pixel stays put."""
    matrix = cv2.getRotationMatrix2D((15.5, 15.5), float(angle), float(scale))
    return cv2.warpAffine(
        query,
        matrix,
        (32, 32),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=float(np.median(query)),
    )


def _parabola_shift(left, center, right):
    denom = left - 2.0 * center + right
    if abs(denom) < 1e-8:
        return 0.0
    shift = 0.5 * (left - right) / denom
    return float(np.clip(shift, -0.75, 0.75))


def _subpixel_center(response, px, py):
    """Centre of the 32x32 window, nudged by a 1D parabola on the correlation peak."""
    height, width = response.shape
    dx = dy = 0.0
    if 0 < px < width - 1:
        dx = _parabola_shift(response[py, px - 1], response[py, px], response[py, px + 1])
    if 0 < py < height - 1:
        dy = _parabola_shift(response[py - 1, px], response[py, px], response[py + 1, px])
    return px + 16.0 + dx, py + 16.0 + dy


def search_patch(sky, query, angles=ANGLES, scales=SCALES, keep=6):
    """Return up to `keep` (ncc, x, y, angle, scale) hypotheses, best first."""
    query = np.ascontiguousarray(query, dtype=np.float32)
    sky = np.ascontiguousarray(sky, dtype=np.float32)
    found = []
    kernel = np.ones((31, 31), np.float32)
    for scale in scales:
        for angle in angles:
            warped = rotate_scale(query, angle, scale)
            response = cv2.matchTemplate(sky, warped, cv2.TM_CCOEFF_NORMED)
            peak = float(response.max())
            if peak < 0.40:
                continue
            dilated = cv2.dilate(response, kernel)
            ys, xs = np.nonzero((response == dilated) & (response >= peak - 0.045))
            if len(xs) == 0:
                continue
            order = np.argsort(-response[ys, xs])[:2]
            for index in order:
                px = int(xs[index])
                py = int(ys[index])
                x, y = _subpixel_center(response, px, py)
                found.append((float(response[py, px]), float(x), float(y), int(angle), float(scale)))

    found.sort(reverse=True)
    kept = []
    for hypothesis in found:
        _, x, y, _, _ = hypothesis
        if all(np.hypot(x - other[1], y - other[2]) > MERGE_PX for other in kept):
            kept.append(hypothesis)
        if len(kept) >= keep:
            break
    return kept


def choose_location(hypotheses):
    """Present/absent from the correlation margin between distinct locations."""
    if not hypotheses:
        return False, None
    best = hypotheses[0]
    rival = None
    for hypothesis in hypotheses[1:]:
        if np.hypot(hypothesis[1] - best[1], hypothesis[2] - best[2]) > MERGE_PX:
            rival = hypothesis
            break
    margin = best[0] - (rival[0] if rival is not None else 0.0)
    present = best[0] >= NCC_FORCE or (best[0] >= NCC_PRESENT and margin >= NCC_MARGIN)
    return present, best


def load_pattern_nodes(path):
    image = np.array(Image.open(path).convert("RGBA"))
    rgb = image[:, :, :3]
    white = (rgb[:, :, 0] > 200) & (rgb[:, :, 1] > 200) & (rgb[:, :, 2] > 200)
    count, _, stats, centroids = cv2.connectedComponentsWithStats(white.astype(np.uint8), 8)
    nodes = [
        centroids[i]
        for i in range(1, count)
        if stats[i, cv2.CC_STAT_AREA] >= 4
    ]
    if not nodes:
        return np.zeros((0, 2), np.float32)
    return np.asarray(nodes, np.float32)


def load_patterns(patterns_dir):
    patterns = {}
    for path in sorted(Path(patterns_dir).glob("*_pattern.png")):
        name = path.stem.replace("_pattern", "")
        nodes = load_pattern_nodes(path)
        if len(nodes) >= 3:
            patterns[name] = nodes
    return patterns


def _owner_inliers(distances, owners, tolerance):
    """Greedy one-to-one match. Each patch owner and each pattern node is used once."""
    order = np.argsort(distances, axis=None)
    used_owner = set()
    used_node = set()
    chosen = []
    errors = []
    n_sky, n_nodes = distances.shape
    for flat in order:
        sky_i, node_i = np.unravel_index(int(flat), distances.shape)
        gap = float(distances[sky_i, node_i])
        if gap > tolerance:
            break
        owner = int(owners[sky_i])
        if owner in used_owner or int(node_i) in used_node:
            continue
        used_owner.add(owner)
        used_node.add(int(node_i))
        chosen.append(int(sky_i))
        errors.append(gap)
    error = float(np.mean(errors)) if errors else 1e9
    return chosen, error


def _fit_similarity(pattern, sky, owners, tolerance, reflect):
    """Geometric-hash similarity. Each patch owner may contribute one inlier.

    Every pair of recovered points defines a frame. Pattern pairs vote for the
    frames that put the rest of the drawing on top of other recovered points.
    """
    sky = np.asarray(sky, np.float32)
    owners = np.asarray(owners)
    source = np.array(pattern, np.float32, copy=True)
    if reflect:
        source[:, 0] *= -1.0
    n_dst = len(sky)
    n_src = len(source)
    if n_src < 3 or n_dst < 3:
        return 0, 1e9, None, []

    table = defaultdict(list)
    bases = []
    step = 1 if n_dst <= 45 else 2
    for i in range(0, n_dst, step):
        for j in range(0, n_dst, step):
            if i == j or owners[i] == owners[j]:
                continue
            edge = sky[j] - sky[i]
            length = float(np.hypot(edge[0], edge[1]))
            if length < 40.0:
                continue
            angle = float(np.arctan2(edge[1], edge[0]))
            cosine, sine = np.cos(angle), np.sin(angle)
            delta = sky - sky[i]
            x_norm = (cosine * delta[:, 0] + sine * delta[:, 1]) / length
            y_norm = (-sine * delta[:, 0] + cosine * delta[:, 1]) / length
            basis_id = len(bases)
            bases.append((i, length, angle))
            for t in range(n_dst):
                if owners[t] == owners[i]:
                    continue
                key = (int(np.floor(x_norm[t] / 0.08)), int(np.floor(y_norm[t] / 0.08)))
                table[key].append(basis_id)
    if not bases:
        return 0, 1e9, None, []

    best_count, best_error, best_pred, best_idx = 0, 1e9, None, []
    src_step = 1 if n_src <= 16 else 2
    for a in range(0, n_src, src_step):
        for b in range(0, n_src, src_step):
            if a == b:
                continue
            edge = source[b] - source[a]
            length = float(np.hypot(edge[0], edge[1]))
            if length < 8.0:
                continue
            angle = float(np.arctan2(edge[1], edge[0]))
            cosine, sine = np.cos(angle), np.sin(angle)
            delta = source - source[a]
            x_norm = (cosine * delta[:, 0] + sine * delta[:, 1]) / length
            y_norm = (-sine * delta[:, 0] + cosine * delta[:, 1]) / length
            votes = defaultdict(int)
            for t in range(n_src):
                bx = int(np.floor(x_norm[t] / 0.08))
                by = int(np.floor(y_norm[t] / 0.08))
                seen = set()
                for dx in (-1, 0, 1):
                    for dy in (-1, 0, 1):
                        for basis_id in table.get((bx + dx, by + dy), ()):
                            if basis_id not in seen:
                                votes[basis_id] += 1
                                seen.add(basis_id)
            if not votes:
                continue
            for basis_id in sorted(votes, key=votes.get, reverse=True)[:3]:
                i, sky_len, sky_ang = bases[basis_id]
                scale = sky_len / length
                dang = sky_ang - angle
                cc, ss = np.cos(dang), np.sin(dang)
                rotation = np.array([[cc, -ss], [ss, cc]], np.float32)
                predicted = sky[i] + scale * ((source - source[a]) @ rotation.T)
                distances = cdist(sky, predicted)
                chosen, error = _owner_inliers(distances, owners, tolerance)
                count = len(chosen)
                if count > best_count or (count == best_count and error < best_error):
                    best_count, best_error = count, error
                    best_pred, best_idx = predicted, chosen
    return best_count, best_error, best_pred, best_idx


def identify(patterns, points, owners=None, tolerance=PATTERN_TOL, min_inliers=MIN_INLIERS):
    """Name the constellation. Owners let one patch offer several locations."""
    points = np.asarray(points, np.float32)
    if owners is None:
        owners = np.arange(len(points))
    else:
        owners = np.asarray(owners)
    if len(points) < 3:
        return "unknown", np.array([], dtype=int), None

    ranked = []
    for name, nodes in patterns.items():
        forward = _fit_similarity(nodes, points, owners, tolerance, reflect=False)
        mirrored = _fit_similarity(nodes, points, owners, tolerance, reflect=True)
        chosen = mirrored if (mirrored[0], -mirrored[1]) > (forward[0], -forward[1]) else forward
        count, error, predicted, indices = chosen
        ranked.append((count, -error, name, predicted, indices))
    ranked.sort(reverse=True)
    count, _, name, predicted, indices = ranked[0]
    if predicted is None or count < min_inliers:
        return "unknown", np.array([], dtype=int), None
    return name, np.asarray(indices, dtype=int), predicted


def scene_prediction(patch_names, hypotheses, patterns):
    """Report the best correlation peak, then name the figure those points form."""
    results = {name: (-1, -1, -1) for name in patch_names}
    chosen = []
    for index, hyps in enumerate(hypotheses):
        present, best = choose_location(hyps)
        if present and best is not None:
            chosen.append(index)

    constellation = "unknown"
    on_figure = set()
    if len(chosen) >= 3:
        points = np.array(
            [[hypotheses[i][0][1], hypotheses[i][0][2]] for i in chosen],
            np.float32,
        )
        constellation, indices, _ = identify(patterns, points)
        on_figure = {chosen[int(i)] for i in indices}

    for index in chosen:
        hypothesis = hypotheses[index][0]
        membership = 1 if index in on_figure else 0
        results[patch_names[index]] = (
            int(round(hypothesis[1])),
            int(round(hypothesis[2])),
            membership,
        )
    return results, constellation


def localization_reward(distance):
    if distance <= 12.0:
        return 1.0
    if distance >= 36.0:
        return 0.0
    return (36.0 - distance) / 24.0


def f1_binary(tp, fp, fn):
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def score_scene(truth, prediction, constellation_hat, constellation_true):
    """One scene under the published weighted metric."""
    present_tp = present_fp = present_fn = 0
    absent_tp = absent_fp = absent_fn = 0
    loc_rewards = []
    figure_true = []
    predicted_present = []

    for name, true in truth.items():
        pred = prediction.get(name, (-1, -1, -1))
        pred_present = pred[2] != -1 and pred[0] != -1
        # Kaggle matches figure stars against ALL reported-present points,
        # including false-positive queries and regardless of the m flag.
        if pred_present:
            predicted_present.append((pred[0], pred[1]))
        if true is None:
            if pred_present:
                # Called a missing patch present.
                present_fp += 1
                absent_fn += 1
            else:
                absent_tp += 1
        else:
            if pred_present:
                present_tp += 1
                distance = float(np.hypot(pred[0] - true[0], pred[1] - true[1]))
                loc_rewards.append(localization_reward(distance))
            else:
                # Missed a patch that is in the sky.
                present_fn += 1
                absent_fp += 1
                loc_rewards.append(0.0)
            if true[2] == 1:
                figure_true.append((true[0], true[1]))

    presence = 0.5 * (
        f1_binary(present_tp, present_fp, present_fn)
        + f1_binary(absent_tp, absent_fp, absent_fn)
    )
    localization = float(np.mean(loc_rewards)) if loc_rewards else 0.0

    if figure_true:
        if predicted_present:
            distances = cdist(np.asarray(figure_true, np.float32), np.asarray(predicted_present, np.float32))
            # nearest pairs first, one to one
            order = np.argsort(distances, axis=None)
            used_f, used_p = set(), set()
            rewards = []
            for flat in order:
                fi, pi = np.unravel_index(flat, distances.shape)
                if fi in used_f or pi in used_p:
                    continue
                used_f.add(int(fi))
                used_p.add(int(pi))
                rewards.append(localization_reward(float(distances[fi, pi])))
                if len(used_f) == len(figure_true):
                    break
            # unmatched figure stars contribute nothing; the mean is over all of them
            geometric = float(np.sum(rewards) / len(figure_true))
        else:
            geometric = 0.0
    else:
        geometric = 0.0

    identification = 1.0 if constellation_hat == constellation_true else 0.0
    total = 0.25 * presence + 0.20 * localization + 0.25 * geometric + 0.30 * identification
    return {
        "presence": presence,
        "localization": localization,
        "geometric": geometric,
        "identification": identification,
        "total": total,
    }


# ---------------------------------------------------------------------------
# Scene runners
# ---------------------------------------------------------------------------

_WORKER_SKY = None


def _init_worker(sky_path):
    global _WORKER_SKY
    cv2.setNumThreads(1)
    _WORKER_SKY = load_gray(sky_path).astype(np.float32)


def _search_worker(query):
    return search_patch(_WORKER_SKY, query)


def list_patches(patch_dir):
    paths = sorted(Path(patch_dir).glob("patch_*.png"))
    return [(path.stem, load_gray(path)) for path in paths]


def search_scene(sky_path, patch_dir, workers=6, cache_path=None):
    import pickle

    cache_path = Path(cache_path) if cache_path else None
    if cache_path is not None and cache_path.exists():
        with cache_path.open("rb") as handle:
            return pickle.load(handle)

    patches = list_patches(patch_dir)
    names = [name for name, _ in patches]
    images = [image for _, image in patches]
    if workers <= 1:
        sky = load_gray(sky_path).astype(np.float32)
        hypotheses = [search_patch(sky, image) for image in images]
    else:
        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=_init_worker,
            initargs=(str(sky_path),),
        ) as pool:
            hypotheses = list(pool.map(_search_worker, images, chunksize=1))
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with cache_path.open("wb") as handle:
            pickle.dump((names, hypotheses), handle)
    return names, hypotheses


def predict_folder(scene_dir, patterns, workers=6):
    scene_dir = Path(scene_dir)
    sky_name = scene_dir.name + "_image.png"
    names, hypotheses = search_scene(scene_dir / sky_name, scene_dir / "patches", workers=workers)
    return scene_prediction(names, hypotheses, patterns)


def format_row(scene_id, n_patches, results, constellation, n_columns=87):
    row = {"Id": scene_id, "n_patches": int(n_patches)}
    for index in range(1, n_columns + 1):
        key = f"patch_{index:02d}"
        if index <= n_patches and key in results:
            x, y, m = results[key]
            row[key] = "-1" if m == -1 or x < 0 else f"({x}, {y}, {m})"
        else:
            row[key] = "-1"
    row["constellation"] = constellation
    return row


def evaluate_training(patterns, workers=6):
    gt = pd.read_csv(ROOT / "train_ground_truth.csv")
    rows = []
    for _, record in gt.iterrows():
        scene = record["Id"]
        print(f"\nSearching train/{scene} ...", flush=True)
        scene_dir = ROOT / "train" / scene
        names, hypotheses = search_scene(
            scene_dir / f"{scene}_image.png",
            scene_dir / "patches",
            workers=workers,
            cache_path=Path(__file__).resolve().parent / "cache" / f"train_{scene}.pkl",
        )
        # Diagnostics use the raw top hypothesis, before pattern snapping.
        truth = {}
        n_patches = int(record["n_patches"])
        for index in range(1, n_patches + 1):
            truth[f"patch_{index:02d}"] = parse_truth(record[f"patch_{index:02d}"])
        print(f"{'patch':8} {'gt':6} {'ncc':7} {'margin':7} {'dist':8} decision")
        for name, hyps in zip(names, hypotheses):
            present, best = choose_location(hyps)
            true = truth.get(name)
            if best is None:
                ncc, margin, dist = 0.0, 0.0, None
            else:
                ncc = best[0]
                rival = next(
                    (h for h in hyps[1:] if np.hypot(h[1] - best[1], h[2] - best[2]) > MERGE_PX),
                    None,
                )
                margin = ncc - (rival[0] if rival else 0.0)
                dist = None if true is None else float(np.hypot(best[1] - true[0], best[2] - true[1]))
            gt_label = "abs" if true is None else f"m{true[2]}"
            dist_text = "" if dist is None else f"{dist:6.1f}"
            flag = "KEEP" if present else "drop"
            ok = ""
            if true is None:
                ok = "TN" if not present else "FP"
            else:
                ok = "TP" if present and dist is not None and dist <= 12 else ("FN" if not present else "LOC")
            print(f"{name:8} {gt_label:6} {ncc:7.3f} {margin:7.3f} {dist_text:>8} {flag:4} {ok}")

        prediction, constellation = scene_prediction(names, hypotheses, patterns)
        metrics = score_scene(truth, prediction, constellation, record["constellation"])
        print(
            f"{scene}: predicted {constellation:16} "
            f"P {metrics['presence']:.3f} L {metrics['localization']:.3f} "
            f"G {metrics['geometric']:.3f} I {metrics['identification']:.0f} "
            f"total {metrics['total']:.3f}"
        )
        rows.append(metrics)
    mean_total = float(np.mean([row["total"] for row in rows]))
    print(f"\nMean training score: {mean_total:.3f}")
    return rows


def write_validation_submission(patterns, path, workers=6):
    sample = pd.read_csv(ROOT / "sample_submission.csv")
    rows = []
    for _, record in sample.iterrows():
        scene = record["Id"]
        print(f"Searching validation/{scene} ...", flush=True)
        scene_dir = ROOT / "validation" / scene
        names, hypotheses = search_scene(
            scene_dir / f"{scene}_image.png",
            scene_dir / "patches",
            workers=workers,
            cache_path=Path(__file__).resolve().parent / "cache" / f"val_{scene}.pkl",
        )
        prediction, constellation = scene_prediction(names, hypotheses, patterns)
        rows.append(format_row(scene, int(record["n_patches"]), prediction, constellation))
        n_present = sum(value != "-1" for key, value in rows[-1].items() if key.startswith("patch_"))
        print(f"  {constellation:16} present patches {n_present}")
    frame = pd.DataFrame(rows, columns=sample.columns)
    frame.to_csv(path, index=False)
    print(f"Wrote {path}")
    return frame


def main():
    patterns = load_patterns(ROOT / "patterns")
    print(f"Loaded {len(patterns)} patterns")
    workers = 6
    evaluate_training(patterns, workers=workers)
    write_validation_submission(
        patterns,
        Path(__file__).resolve().parent / "submission.csv",
        workers=workers,
    )


if __name__ == "__main__":
    main()
