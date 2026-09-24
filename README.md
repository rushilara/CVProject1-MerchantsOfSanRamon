# Constellation detection — deadline workflow

Use the existing completed correlation caches. The default notebook and Slurm scripts run **cached submission only**, without another correlation search or synthetic parameter sweep.

## NYU Torch

Use `/scratch/ra4880/q5const` and its existing `cache_hpc/`. Copy the updated Python files and `torch_cpu.slurm` there before submitting. This repository update does not update or restart an already running cluster job.

```bash
cd /scratch/ra4880/q5const
mkdir -p logs
sbatch --account=torch_pr_37_lpinto --partition=cs torch_cpu.slurm
```

The script validates all 16 validation cache headers, patch lists and source files, then runs `--submit`. It uses the defaults in `PARAMS`; an old `best_hpc.pkl` is no longer loaded automatically. Do not launch `--all`, `--sweep`, or `--search-all` for this deadline workflow.

Each completed scene is saved under `submission_hpc.checkpoints/`. If interrupted, rerun the same command to reuse completed predictions. Checkpoints are invalidated when code, prediction parameters, pattern nodes or input-file metadata change. Keep code and inputs unchanged during a run; do not run two writers against the same output path concurrently.

The complete `submission_hpc.csv` is replaced atomically only after every scene succeeds. An older CSV may remain after a failed run, so wait for the explicit `wrote complete 16-scene submission` message before uploading.

## Local or notebook execution

`Computer_Vision_Project_1_Question_5_HPC.ipynb` follows the same cached-only workflow. It requires the completed caches; cloning the repository does not supply them. Run on an allocated compute node, not an HPC login node.

```bash
# With the completed fine caches:
Q5_FINE=1 Q5_CACHE=/path/to/cache_hpc python -u q5_hpc.py --smoke
Q5_FINE=1 Q5_CACHE=/path/to/cache_hpc python -u q5_hpc.py --submit
```

Without overrides, local execution uses the available coarse `cache_v3` first. Coarse and fine runs have different search evidence and proposal settings, so their scores and timings are not interchangeable. An explicitly supplied `Q5_CACHE` is authoritative: missing or malformed caches stop before inference and never trigger a search.

Dependencies: `pip install -r requirements-hpc.txt`. This pipeline uses CPU OpenCV/NumPy/SciPy, not a GPU. Cached submission currently processes scenes sequentially; the search-worker count does not parallelize it. Full fine-cache runtime has not been measured by this review.

## Kaggle metric

The supplied competition description specifies the scene-wise mean of:

- 25% presence: macro F1 for present versus absent.
- 20% localization: mean over truly present queries; full credit within 12 px, linear decay to zero at 36 px; missed queries earn zero.
- 25% geometric recovery: greedy one-to-one nearest-pair matching of issued figure stars to all reported-present points, using the same distance reward. This term does not filter by predicted `m`.
- 30% identification: exact constellation name.

`q5_constellation.score_scene` follows this description. Scenes have equal weight regardless of patch count. Reference drawings may differ in aspect ratio as well as orientation, scale and handedness; the existing similarity-only proposal model does not fully cover this requirement. The brightest-120-star filter also excludes most labeled figure stars. Do not infer generalization from tuning on only three training scenes.

## Fast regression checks

```bash
OPENBLAS_NUM_THREADS=1 python -m unittest test_q5_hpc -v
```

These check exact claim-map values, interrupted-run resume, checkpoint invalidation and missing/mismatched cache rejection. They do not perform the expensive search or prove leaderboard performance.
