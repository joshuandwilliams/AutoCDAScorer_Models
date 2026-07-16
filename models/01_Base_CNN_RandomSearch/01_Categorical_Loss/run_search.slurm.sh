#!/bin/bash
#SBATCH -p jic-compute
#SBATCH --cpus-per-task=8
#SBATCH --mem 16000
#SBATCH --time=0-04:00:00
#SBATCH --job-name="base_cnn_search"
#SBATCH -o slurm.run_search_%a.out
#SBATCH -e slurm.run_search_%a.err
#SBATCH --array=1-2
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=jowillia@nbi.ac.uk

# Runs on the CPU queue (jic-compute): the models are tiny and the GPU queue can
# pend for a long time. The GPU-built container runs fine on CPU -- TensorFlow just
# logs a harmless "no GPU found" note and falls back to CPU (no --nv / --gres here).
#
# Small CHECK run: 2 array tasks (seeds 1 and 2), each training 4 models and keeping
# artefacts for only its top 2 -- enough to verify the per-task output folders and
# the storage-efficient top-N logic. For a real sweep, raise --array, n_models, and
# top_n (top_n back to ~100).
img="$HOME/singularity/TensorFlow/TensorFlowGPU_2_21_0.img"
n_models=4
top_n=2

singularity exec ${img} python3.12 run_search.py --slurm_array $SLURM_ARRAY_TASK_ID --n_models ${n_models} --top_n ${top_n}

# Tidy the SLURM logs into this task's output folder (created by the run).
mkdir -p array_task${SLURM_ARRAY_TASK_ID}
mv slurm.run_search_${SLURM_ARRAY_TASK_ID}.out array_task${SLURM_ARRAY_TASK_ID}/ 2>/dev/null || true
mv slurm.run_search_${SLURM_ARRAY_TASK_ID}.err array_task${SLURM_ARRAY_TASK_ID}/ 2>/dev/null || true
