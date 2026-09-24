"""Fast regression checks; no correlation search or full figure identification."""
import json
import os
import pickle
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from PIL import Image
from scipy.ndimage import maximum_filter
import q5_hpc as q


class MetricTests(unittest.TestCase):
    def test_figure_recovery_includes_false_positive_queries_and_ignores_m(self):
        truth = {"patch_01": (50, 50, 1), "patch_02": None}
        prediction = {"patch_01": (-1, -1, -1), "patch_02": (50, 50, 0)}
        score = q.score_scene(truth, prediction, "wrong", "test")
        self.assertEqual(score["presence"], 0.)
        self.assertEqual(score["localization"], 0.)
        self.assertEqual(score["geometric"], 1.)
        self.assertEqual(score["total"], 0.25)

    def test_one_point_cannot_recover_two_figure_stars(self):
        truth = {"a": (50, 50, 1), "b": (52, 50, 1)}
        score = q.score_scene(truth, {"a": (50, 50, 0)}, "test", "test")
        self.assertEqual(score["geometric"], 0.5)


class CachedSubmissionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.cache = self.root / "cache"
        self.cache.mkdir()
        rows = []
        for scene in ("scene_a", "scene_b"):
            folder = self.root / "validation" / scene
            (folder / "patches").mkdir(parents=True)
            Image.fromarray(np.zeros((64, 64), np.uint8)).save(folder / f"{scene}_image.png")
            Image.fromarray(np.zeros((32, 32), np.uint8)).save(folder / "patches/patch_01.png")
            with (self.cache / f"val_{scene}.pkl").open("wb") as handle:
                pickle.dump((["patch_01"], [[(0.95, 32., 32., 0, 1.)]]), handle)
            np.save(self.cache / f"val_{scene}_maps.npy", np.zeros((1, 17, 17), np.float16))
            rows.append(q.format_row(scene, 1, {}, "unknown"))
        pd.DataFrame(rows).to_csv(self.root / "sample_submission.csv", index=False)
        self.out = self.root / "submission.csv"
        for ctx in (patch.object(q, "ROOT", self.root), patch.object(q, "CACHE", self.cache),
                    patch.dict(os.environ, {"Q5_CACHE": str(self.cache)})):
            ctx.start()
            self.addCleanup(ctx.stop)
        self.patterns = {"test": np.array([[0., 0.], [1., 1.], [0., 2.]])}

    def test_claims_bitwise_equivalent(self):
        maps = np.random.default_rng(0).uniform(-1, 1, (7, 31, 29)).astype(np.float16)
        for radius in (0, 2):
            old = np.stack([maximum_filter(m.astype(np.float32), size=2*radius+1,
                                         mode="nearest") for m in maps]).astype(np.float16)
            claims = q.Claims(maps, radius, 2)
            np.testing.assert_array_equal(claims.dil, old)
            np.testing.assert_array_equal(claims.best, old.max(axis=0).astype(np.float32))

    def test_interrupted_run_resumes_and_settings_invalidate(self):
        result = ("test", {}, [])
        with patch.object(q, "figure_stage", side_effect=[result, RuntimeError("interrupted")]), \
             patch.object(q, "search_scene", side_effect=AssertionError("must not search")):
            with self.assertRaisesRegex(RuntimeError, "interrupted"):
                q.write_submission(self.patterns, self.out)
        self.assertFalse(self.out.exists())
        self.assertTrue(self.out.with_suffix(".checkpoints").joinpath("scene_a.json").exists())
        with patch.object(q, "figure_stage", return_value=result) as stage:
            q.write_submission(self.patterns, self.out)
            self.assertEqual(stage.call_count, 1)
        df = pd.read_csv(self.out)
        self.assertEqual(list(df.Id), ["scene_a", "scene_b"])
        self.assertEqual(df.loc[0, "patch_01"], "(32, 32, 0)")
        with patch.object(q, "figure_stage", side_effect=AssertionError("should resume")):
            q.write_submission(self.patterns, self.out)
        with patch.object(q, "figure_stage", return_value=result) as stage:
            q.write_submission(self.patterns, self.out, dict(q.PARAMS, high=0.89))
            self.assertEqual(stage.call_count, 2)

    def test_missing_cache_fails_before_inference_or_search(self):
        (self.cache / "val_scene_b.pkl").unlink()
        with patch.object(q, "figure_stage") as stage, patch.object(q, "search_scene") as search:
            with self.assertRaisesRegex(RuntimeError, "refusing"):
                q.write_submission(self.patterns, self.out)
            stage.assert_not_called()
            search.assert_not_called()

    def test_bad_patch_names_rejected(self):
        with (self.cache / "val_scene_b.pkl").open("wb") as handle:
            pickle.dump((["patch_99"], [[]]), handle)
        with self.assertRaisesRegex(ValueError, "patch names"):
            q.validate_caches(pd.read_csv(self.root / "sample_submission.csv"))

    def test_changed_cache_invalidates_only_affected_scene(self):
        with patch.object(q, "figure_stage", return_value=("test", {}, [])) as stage:
            q.write_submission(self.patterns, self.out)
            stage.reset_mock()
            path = self.cache / "val_scene_a_maps.npy"
            st = path.stat()
            os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns + 1000000))
            q.write_submission(self.patterns, self.out)
            self.assertEqual(stage.call_count, 1)


if __name__ == "__main__":
    unittest.main()
