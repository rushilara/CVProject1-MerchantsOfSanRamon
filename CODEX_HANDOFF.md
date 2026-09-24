# Constellation pipeline and NYU Torch

Handoff for the Kaggle constellation task. The cluster copy that jobs actually run is `/scratch/ra4880/q5const` on Torch. The laptop repo is the same code. Do not rerun the fine search or the 24-config sweep.

## What we are implementing

Each scene is a 3000×3000 star field plus 32×32 patches. A patch is centered on a star in that field, or it comes from somewhere else. For every patch the submission reports `(x, y, m)` or `-1`, plus one constellation name.

Score per scene:

`0.25 * presence + 0.20 * localization + 0.25 * figure recovery + 0.30 * constellation name`

The name and figure recovery are 55% of the score. A previous Kaggle submission scored **0.306** because the figure was often never proposed. Bright-star patches correlate at 0.95+ with many other bright stars, and some figure neighborhoods are pasted elsewhere as pixel copies. Taking the best normalized cross-correlation peak, or fitting only the top few drawings, named the wrong constellation and missed the figure.

The pipeline in `q5_hpc.py` is classical computer vision. There is no neural net.

1. **Search (already done).** Each patch is warped through 72 angles (every 5°) and 7 scales and correlated against the sky with `cv2.matchTemplate`. We keep up to 24 distinct peaks and a coarse map of the best correlation at every sky position (`cache_hpc/*_maps.npy`, pooled by 2). This is the expensive stage. It is finished for all 35 scenes.
2. **Name the figure.** Detect bright stars independently of the patches. Propose a similarity transform, including reflection, for every one of the 48 pattern drawings, using both the star catalog (geometric hash in `q5_identify.py`) and near-tied patch peaks. Verify each proposal by a Hungarian assignment of patches to predicted nodes on the claim maps. Rank drawings by how many nodes are claimed above chance for a drawing of that size (`rule=excess`, `select=residual`), so a large drawing such as Hydra cannot win just by offering more targets.
3. **Write coordinates.** Claimed nodes are reported as `m = 1`. Other patches are present (`m = 0`) when the best peak is at least 0.90, or at least 0.55 and 0.04 above the next distinct location. Otherwise `-1`. If the winning drawing is clearly ahead, leftover nodes can be filled by leftover patches that still claim them (`spare_fill`). Figure recovery matches predicted points to true figure stars, not patch identity.

Settings already measured on the partial sweep, and currently the defaults in `PARAMS`:

- `rule=excess`
- `select=residual`
- `high=0.88`
- `spare_fill=True`

On the 3 real training scenes that setting scored **0.774** and named **3/3** correctly. Synthetic scenes scored about **0.49** with **0/16** names. Synthetic identification is unsolved. Do not retune on it tonight unless the submission file is already written. The Kaggle file only needs the 16 validation scenes.

## What is running

| Item | Value |
| --- | --- |
| Job | `18424834` (confirm with `squeue`; it was pending on `Priority`) |
| Script | `/scratch/ra4880/q5const/torch_cpu.slurm` |
| Work dir | `/scratch/ra4880/q5const` |
| Partition / account | `cs` / `torch_pr_37_lpinto` |
| Resources | 1 node, 1 task, 32 CPUs, 48 GB, 8 hours |
| Log | `/scratch/ra4880/q5const/logs/q5const_18424834.out` |
| Error log | `/scratch/ra4880/q5const/logs/q5const_18424834.err` |
| Output we need | `/scratch/ra4880/q5const/submission_hpc.csv` |

The script runs `python -u q5_hpc.py --smoke` and, only if that exits 0, `python -u q5_hpc.py --submit`.

`--smoke` is a few seconds. It checks that all 16 `val_*` caches exist, loads 8 maps from the smallest one, and builds one claim stack. It does not identify a constellation. A full scene would cost about as much as 1/16 of the real submission, which is a bad pretest.

`--submit` reads the caches and identifies each validation scene once. It does not call `--all` and does not sweep.

Search cache, measured after the failed job:

- Path: `/scratch/ra4880/q5const/cache_hpc`
- Size: 5.9 GB
- 16 validation + 3 train + 16 synthetic scenes
- Every `*_maps.npy` has a matching `*.pkl`

Python environment: `/scratch/ra4880/q5const/.venv` (OpenCV, NumPy, SciPy, pandas, Pillow).

## Failures already paid for

Do not repeat these.

**Job 18358669, TIMEOUT, 8 hours.** `fork` after importing OpenCV hung the process pool. RSS stayed about 300 MB, so the 32 workers never started. One process ground through `train_pisces` until the wall clock. Fixed by `multiprocessing.get_context("spawn")` in `search_folder`. The search has since completed. Do not launch another `--search-all` unless a cache file is missing.

**Job 18399338, OUT_OF_MEMORY, 2 hours 25 minutes, peak 48 GB.** `q5_hpc.py --all` ran the search (cached afterward) and then a 24-config sweep that kept every configuration's dilated claim maps. Slurm killed `python -u q5_hpc.py --all`. The sweep no longer stores those arrays, but do not run `--all` or `--sweep` for tonight's file. The useful setting is already the default.

Parallelism is one process per patch, not threads inside a process. `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `NUMEXPR_NUM_THREADS`, and `OPENCV_NUM_THREADS` are all 1 on purpose. Setting them to 32 would start 32 threads inside each of 32 processes. `--ntasks=1 --cpus-per-task=32` matches the process pool. The largest scene has 87 patches, so 32 workers is enough. 48 GB is enough for `--submit` because only one scene's claim stack is live (largest stack is about 0.4 GB). The 48 GB death was the sweep holding many scenes at once.

## Torch: how to use it

Official docs: https://services.rt.nyu.edu/

Cluster name is Torch. This account is NetID `ra4880`.

### Connect

VPN or the NYU network is required.

Laptop SSH config already has:

```sshconfig
Host torch
  Hostname login.torch.hpc.nyu.edu
  User ra4880
```

```bash
ssh torch
```

SSH keys are disabled. The server prints a one-time PIN. Open https://login.microsoft.com/device, enter the PIN, sign in with the NYU account, then press Enter in the SSH session. A new PIN is required for each new SSH connection. `squeue` on the laptop will fail. Run Slurm commands only after `ssh torch`.

Login nodes are `torch-login-*`, Red Hat. Do not run the correlation search or a long Python job on a login node. Submit with `sbatch`.

### Where the files are

| Path | What |
| --- | --- |
| `/home/ra4880` | Home. Small. Do not put caches here. |
| `/scratch/ra4880/q5const` | This project, the venv, caches, and logs. |
| `dtn.torch.hpc.nyu.edu` | Data-transfer node for `rsync` of large files. |

Copy a changed file from the laptop:

```bash
rsync -av q5_hpc.py torch_cpu.slurm torch:/scratch/ra4880/q5const/
```

`rsync` to `torch` opens the same device-login prompt unless an SSH control socket is already open.

### Slurm for this project

Every job needs an account. This user has `torch_pr_37_lpinto` (the one to use) and `users`.

```bash
cd /scratch/ra4880/q5const
sbatch --account=torch_pr_37_lpinto --partition=cs torch_cpu.slurm
```

`cs` is the CPU-only partition: 184 nodes, 128 cores, about 513 GB RAM, no GPU, preemption off. Requesting no GPU and not setting a partition can still land a CPU job on a GPU node, where a GPU job can preempt it. For this pipeline, set `--partition=cs`.

Do not request a GPU. The code never uses one. Torch cancels GPU jobs that stay below a utilization threshold after 2 hours (for example 50% on L40S, 60% on H100/H200). GPU requests also cap CPUs at about 16 per GPU.

A wall time under 48 hours is placed in QOS `cpu48` automatically. `cpu_short` style limits are not the binding constraint on `cs`.

Useful commands, run on a login node:

```bash
squeue -u ra4880
squeue -j JOBID
scontrol show job JOBID
sacct -j JOBID --format=JobID,State,ExitCode,Elapsed,Start,End,MaxRSS,AllocCPUS
seff JOBID
tail -f /scratch/ra4880/q5const/logs/q5const_JOBID.out
tail -n 80 /scratch/ra4880/q5const/logs/q5const_JOBID.err
scancel JOBID
```

`squeue` only lists jobs that are still pending or running. `Invalid job id` means the job finished, failed, timed out, or was killed. Use `sacct` for the final state. `PD` is pending. `R` is running. `Priority` means other jobs are ahead, not that the request is impossible. `OUT_OF_MEMORY` is a cgroup kill. `TIMEOUT` is the wall clock.

`#SBATCH --requeue` is set. If the job is preempted it can restart. `--submit` is safe to rerun only when `cache_hpc` is intact, because a missing cache makes stage 1 recompute that scene. `--open-mode=append` means a requeued job appends to the same log.

### What not to change on the cluster

- Do not run `python q5_hpc.py --all` or `--sweep` in the current allocation. That is the OOM.
- Do not delete `/scratch/ra4880/q5const/cache_hpc`.
- Do not set `Q5_FINE=1` together with `force` search unless a specific scene's `.pkl` and `_maps.npy` are missing.
- Do not raise `OMP_NUM_THREADS` to the CPU count.
- Do not add `#SBATCH --gres=gpu`.

### Other Torch facts, if a later job needs them

- Project portal (allocations and Slurm accounts): https://projects.hpc.nyu.edu/ on NYU VPN. Account names look like `torch_pr_<id>_<resource>`.
- Open OnDemand: https://ood.torch.nyu.edu
- Metrics: https://stats.apps.cloud.rt.nyu.edu/
- GPU types include H200, L40S, B200, H100, A100, RTX Pro 6000. Public GPU use is through preemption. A preemptable job adds `#SBATCH --comment="preemption=yes;requeue=true"`. Preemption can cancel the job after 30 minutes. Not used here.
- The scheduler normally picks a GPU partition from the resources you request. The slides say not to set `--partition` yourself for GPU jobs. This CPU job is an exception: `--partition=cs` keeps it off GPU nodes.
- Containers: Apptainer/Singularity images in `/share/apps/images`, overlays in `/share/apps/overlay-fs-ext3`. This project uses a venv instead.
- File ACLs use `nfs4_setfacl` / `nfs4_getfacl`, not `setfacl`.
- Globus collections are under the NYU#Torch label for home, scratch, archive, and project.

## When the job finishes

```bash
sacct -j 18424834 --format=JobID,State,ExitCode,Elapsed,MaxRSS
ls -l /scratch/ra4880/q5const/submission_hpc.csv
```

`State=COMPLETED` and a non-empty `submission_hpc.csv` is the Kaggle upload. Copy it back with `rsync` from `torch:/scratch/ra4880/q5const/submission_hpc.csv`. If the state is `OUT_OF_MEMORY` or `FAILED`, read the `.err` file before changing the resource request. Do not bump memory on a hunch. The `--submit` path was sized so one scene's claim stack is about 0.4 GB.
