#!/bin/bash
# CPU-only constellation run for NYU Torch.
#   sbatch --account=torch_pr_37_lpinto --partition=cs torch_cpu.slurm
# Do not request a GPU. This job is cv2.matchTemplate in a process pool.
#
# Parallelism is one Python process per query patch (ProcessPoolExecutor),
# not OpenMP/MKL inside one process. A scene has 21–87 patches, so 32
# workers keep the large scenes busy without asking for 75% of a node.
# Each worker is pinned to 1 OpenCV/BLAS thread to avoid oversubscribe.

#SBATCH --job-name=q5const
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=logs/q5const_%j.out
#SBATCH --error=logs/q5const_%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"
mkdir -p logs cache_hpc

export Q5_FINE=1
export Q5_WORKERS="${SLURM_CPUS_PER_TASK:-32}"
export Q5_CACHE="${Q5_CACHE:-$PWD/cache_hpc}"
# One thread per process. The 32 cores are the 32 workers, not 32 threads
# inside each worker.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OPENCV_NUM_THREADS=1
export MPLCONFIGDIR="${SLURM_TMPDIR:-/tmp}/mpl-${SLURM_JOB_ID:-local}"

if [[ -f .venv/bin/activate ]]; then
  source .venv/bin/activate
fi

echo "job ${SLURM_JOB_ID:-local} host $(hostname) cpus ${Q5_WORKERS} $(date -Is)"
python - <<'PY'
import importlib.util, os, sys
need = ["cv2", "numpy", "pandas", "scipy", "PIL"]
miss = [n for n in need if importlib.util.find_spec(n) is None]
if miss:
    sys.exit("missing packages: " + ", ".join(miss))
print("python", sys.executable, "workers", os.environ.get("Q5_WORKERS"), "cache", os.environ.get("Q5_CACHE"))
PY

# Validate all cache headers and patch lists, then exercise eight claim planes.
# This does not run identification. Submission checkpoints each completed scene.
echo "smoke $(date -Is)"
python -u q5_hpc.py --smoke
echo "smoke ok $(date -Is)"
python -u q5_hpc.py --submit
echo "done $(date -Is). submission is submission_hpc.csv"
