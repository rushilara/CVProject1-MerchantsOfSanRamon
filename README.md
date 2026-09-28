# Constellation Detection (CS-GY 6643 Project 1, Q5)

Everything lives in **`constellation_detection.ipynb`**. It plate-solves each validation sky against the 48 reference
patterns, decides for every 32 × 32 query tile whether it is present (and where), and names the constellation.

## Layout

```
constellation_detection.ipynb   the pipeline (self-contained)
participant/                    competition data: patterns/, train/, validation/, sample_submission.csv, train_ground_truth.csv
requirements.txt
```

## Running

**Colab (recommended, T4 GPU):** open the notebook and run all cells. It looks for the data at
`/content/drive/MyDrive/constellation_data/participant` (mounting Drive) or `/content/participant`, and writes
`submission.csv` to `/content`.

**Locally:**

```bash
pip install -r requirements.txt
jupyter notebook constellation_detection.ipynb
```

The notebook finds `participant/` next to itself, or wherever `CONST_DATA` points. Output goes to `./submission.csv`.
A CUDA GPU speeds up the correlation sweep; the CPU fallback works but is slower.
