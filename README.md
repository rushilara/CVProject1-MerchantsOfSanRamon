# Constellation detection — Colab / HPC run

This branch is what you run for the Kaggle question. Open the notebook on Colab, clone this branch, run all cells, upload `submission_hpc.csv`.

## What to run

**Notebook:** `Computer_Vision_Project_1_Question_5_HPC.ipynb`

That is the only notebook you need. It calls `q5_hpc.py`. Do not run the older Q5 notebooks on Colab.

## Colab (recommended)

1. Runtime → Change runtime type → **CPU**. Turn on **High-RAM** if you have Colab Pro. Do **not** pick a GPU — the pipeline is `cv2.matchTemplate` on CPU.
2. Upload this notebook to Colab, or open it from the GitHub branch.
3. In the first code cell, keep the clone URL/branch as-is (or set `PROJECT` if you already copied the repo onto Drive).
4. Run all cells in order. Stage 1 (correlation sweep) can take several hours on Colab; results land in `cache_hpc/` so a reconnect can continue.
5. The last cell writes `submission_hpc.csv` and downloads it. Upload that file to Kaggle.

### Packages

The first cell installs these. Nothing else is required (no PyTorch, no TensorFlow, no CUDA):

```
opencv-python-headless
numpy
scipy
pandas
pillow
```

Same list is in `requirements-hpc.txt`.

### Files this notebook needs (all on this branch)

| Path | Why |
| --- | --- |
| `q5_hpc.py` | Search, identify, sweep, write CSV |
| `q5_identify.py` | Pattern nodes, geometric hash, sky support |
| `q5_constellation.py` | Score + submission row format |
| `q5_search.py` | Rotate/scale + sub-pixel peak |
| `q5_synth.py` | Star catalog (and building extra synthetic scenes) |
| `participant/` | Train + validation skies, patches, 48 pattern PNGs, CSVs |
| `synthetic/` | Extra labelled scenes for the rule sweep |

`cache_v3/` is **not** on this branch (it is ~1.5 GB). Leave `Q5_FINE=1` so Colab rebuilds a finer cache under `cache_hpc/`.

## Real HPC node (optional)

```bash
git clone -b colab-hpc git@github.com:rushilara/CVProject1-MerchantsOfSanRamon.git
cd CVProject1-MerchantsOfSanRamon
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-hpc.txt
export Q5_FINE=1 Q5_WORKERS=32
python -u q5_hpc.py --all
# or: sbatch hpc.slurm
```

## What not to commit / upload

Ignored on purpose: `cache_v3/` and other caches, lecture PDFs, the 22 MB Q1–4 notebook, `__pycache__/`, generated `submission_*.csv`.
