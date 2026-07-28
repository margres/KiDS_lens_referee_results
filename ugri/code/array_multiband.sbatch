#!/bin/bash -l
#SBATCH -J kids_multiband
#SBATCH -n 1
#SBATCH --cpus-per-task=8
#SBATCH -o output/output_mb.%A_%a.log
#SBATCH -e output/output_mb.%A_%a.log
#SBATCH --partition=cmb
#SBATCH -t 12:00:00
#SBATCH --mem=16gb
#SBATCH --array=0-222%20

# Multi-band (r+g+i, u excluded -- too noisy, see fit_lens_model_multiband.py
# docstring) fit for the 223 known+new grade A/B objects, chained to start
# automatically after the g/u/i cutout fetch (job 798346) completes via
# --dependency=afterok. Any object missing a g or i cutout (fetch failure
# for that tile) fails its own task cleanly (FileNotFoundError) without
# blocking the rest of the array.

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
export NUMBA_CACHE_DIR="/tmp/kids_lens_numba_mb_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
export MPLCONFIGDIR="/tmp/kids_lens_mpl_mb_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
mkdir -p "$NUMBA_CACHE_DIR" "$MPLCONFIGDIR"

DATASET=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" all_AB_ids.txt)

echo "multiband task $SLURM_ARRAY_TASK_ID -> dataset $DATASET"
date

python3 fit_lens_model_multiband.py --dataset "$DATASET" --bands r,g,i \
    --results_dir results_multiband --number_of_cores $THREADS

echo "Finished $DATASET"
date
rm -rf "$NUMBA_CACHE_DIR" "$MPLCONFIGDIR"
