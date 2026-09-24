#!/bin/bash
# Run on a single HPC node. Adjust the partition / account for your cluster.
#   sbatch hpc.slurm
#
# Cached submission only; use torch_cpu.slurm for NYU Torch.
# No correlation search or parameter sweep is started.
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
export Q5_CACHE="${Q5_CACHE:-$PWD/cache_hpc}"
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
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

python -u q5_hpc.py --smoke
python -u q5_hpc.py --submit
echo "done. submission_hpc.csv is ready to upload."
