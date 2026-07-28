#!/bin/bash -l
#SBATCH -J kids_all_AB
#SBATCH -n 1
#SBATCH --cpus-per-task=8
#SBATCH -o output/output.%A_%a.log
#SBATCH -e output/output.%A_%a.log
#SBATCH --partition=cmb
#SBATCH -t 12:00:00
#SBATCH --mem=16gb
#SBATCH --array=0-222%20

# Full correct target set: the 223 deduplicated known(107)+new(140) grade
# A/B lenses (AB_fits/, all_AB_ids.txt), NOT the old targets.csv
# 107 sample used earlier this session (only 10/107 actually overlapped
# with the real known_lenses_AB.csv ground truth -- see memory). Reads
# cutouts directly from AB_fits/, no copy needed. Mask fix
# (7" default) and --source_re_max 30.0 already baked in / passed.
# PyAutoFit will legitimately skip-and-reuse any of the ~10 objects that
# already have a matching completed result from the earlier (correct by
# coincidence) run -- that's desired here, not the cache-staleness problem
# from the mask-fix rerun (same code, same config, so a real skip is valid).

set -e
PROJECT_DIR="/mnt/users/grespanm/kids_hpc_cluster/hpc"
cd "$PROJECT_DIR"

source /mnt/users/grespanm/venvs/kids_lens/bin/activate
export PYAUTO_SKIP_WORKSPACE_VERSION_CHECK=1

THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$THREADS
export MKL_NUM_THREADS=$THREADS
export OMP_NUM_THREADS=$THREADS
export JAX_PLATFORM_NAME=cpu
export JAX_PLATFORMS=cpu
export NUMBA_CACHE_DIR="/tmp/kids_lens_numba_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
export MPLCONFIGDIR="/tmp/kids_lens_mpl_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
mkdir -p "$NUMBA_CACHE_DIR" "$MPLCONFIGDIR"

DATASET=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" all_AB_ids.txt)

echo "all_AB task $SLURM_ARRAY_TASK_ID -> dataset $DATASET"
date

python3 fit_lens_model.py --dataset "$DATASET" --cutout_dir AB_fits \
    --results_dir results --number_of_cores $THREADS --source_re_max 30.0

echo "Finished $DATASET"
date
rm -rf "$NUMBA_CACHE_DIR" "$MPLCONFIGDIR"
