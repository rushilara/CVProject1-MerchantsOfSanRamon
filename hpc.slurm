#!/bin/bash
# Run on a single HPC node. Adjust the partition / account for your cluster.
#   sbatch hpc.slurm
#
# The job:
#   1. finer 5° × 7-scale correlation sweep for every scene (Q5_FINE=1)
#   2. held-out sweep of figure-stage rules on train + synthetic
#   3. writes submission_hpc.csv
#
#SBATCH --job-name=q5hpc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs/q5hpc_%j.out
#SBATCH --error=logs/q5hpc_%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"
mkdir -p logs

if [[ -n "${SLURM_CPUS_PER_TASK:-}" ]]; then
  export Q5_WORKERS="$SLURM_CPUS_PER_TASK"
else
  export Q5_WORKERS="${Q5_WORKERS:-32}"
fi
export Q5_FINE=1
export OMP_NUM_THREADS=1
export OPENCV_NUM_THREADS=1

python - <<'PY'
import importlib.util, os, sys
need = ["cv2", "numpy", "pandas", "scipy", "PIL"]
miss = [n for n in need if importlib.util.find_spec(n) is None]
if miss:
    sys.exit("missing packages: " + ", ".join(miss) + "\ninstall: pip install -r requirements-hpc.txt")
print("python", sys.executable, "workers", os.environ.get("Q5_WORKERS"))
PY

python -u q5_hpc.py --all
echo "done. submission_hpc.csv is ready to upload."
