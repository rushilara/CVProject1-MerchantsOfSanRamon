# Deadline review, September 24, 2026

The user supplied the Kaggle task and evaluation description during this review. The metric is a mean over scenes of 0.25 presence macro-F1 + 0.20 localization + 0.25 geometric recovery + 0.30 exact-name identification. Localization and recovery use full reward through 12 px and linear decay to zero at 36 px.

## Changes retained

- Corrected local geometric recovery to consider every reported-present point, including points reported for truly absent queries. The old evaluator excluded those points. Predicted membership `m` does not gate geometric recovery in the supplied specification.
- Submission reads existing caches directly and validates all requested scenes before inference; it cannot fall through into correlation search.
- Explicit `Q5_CACHE` takes precedence without fallback. Cache patch lists, dimensions and source-file presence are checked.
- Atomic per-scene JSON checkpoints support restart. Signatures cover prediction code, parameters, pattern nodes and input metadata. Final CSV is atomically replaced only when all requested scenes complete.
- Claim-stack construction now fills one float16 allocation a plane at a time. Regression tests establish bitwise equality with the prior construction.
- Notebook and Slurm workflows no longer run the synthetic sweep. The notebook's obsolete `best[6]` access is removed with that stage.
- Old `best_hpc.pkl` settings are not silently applied to submission.

## Algorithm findings and bounded experiments

The brightest-120-star filter retained only 1/26 labeled figure stars within 12 px on the three real training scenes. The existing patch-proposal cloud retained correct owner-specific candidates for 21/26 figure patches. Globally increasing the star catalog would sharply increase geometric-hash work; no such increase was made.

The description explicitly allows reference aspect ratio differences. The existing model fits similarity transforms and reflection, so it does not fully model those differences. A bounded experiment generated patch-based proposals at horizontal aspect factors 0.5, 0.67, 0.8, 1, 1.25, 1.5 and 2, omitting the independent bright-star proposals. It took about 43 seconds total on the local coarse caches and named 2/3 training scenes correctly. The existing coarse-cache method also named 2/3 correctly, taking about 52 seconds total. Pisces was missed by both. Results did not establish a dependable gain, so this experimental method was not adopted. The synthetic generator also independently snaps nodes by up to 60 px, which is a separate geometry mismatch.

These tests used local `cache_v3`, not the fine `cache_hpc` on Torch. They do not reproduce or contradict the handoff's fine-run score of 0.774. No full fine-cache run, cluster job or Kaggle upload was performed. Prediction thresholds and candidate-generation rules remain unchanged.

## Verification

Seven fast regression tests cover claim-map equivalence, interrupted-run resume, parameter/cache invalidation, missing caches, mismatched patch names, and two geometric-recovery metric cases. All 16 local validation cache headers and patch lists passed preflight. Notebook code and shell syntax checks passed.

Use README.md for the cached-only run commands. Local changes must be transferred to Torch before a new cluster run can use them; an existing job is unaffected.
