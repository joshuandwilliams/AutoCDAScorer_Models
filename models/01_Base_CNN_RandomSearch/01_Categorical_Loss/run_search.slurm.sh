#!/bin/bash
#SBATCH -p jic-compute
#SBATCH --cpus-per-task=8
#SBATCH --mem 16000
#SBATCH --time=0-04:00:00
#SBATCH --job-name="base_cnn_search"
#SBATCH -o slurm.run_search_%a.out
#SBATCH -e slurm.run_search_%a.err
#SBATCH --array=1-1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=jowillia@nbi.ac.uk

# Runs on the CPU queue (jic-compute): the models are tiny and the GPU queue can
# pend for a long time. The GPU-built container runs fine on CPU -- TensorFlow just
# logs a harmless "no GPU found" note and falls back to CPU (no --nv / --gres here).
#
# Small check run: one array task training 4 randomly sampled models. To scale up,
# raise --array (each task uses its index as the RNG seed) and/or n_models.
img="$HOME/singularity/TensorFlow/TensorFlowGPU_2_21_0.img"
n_models=4

singularity exec ${img} python3.12 run_search.py --slurm_array $SLURM_ARRAY_TASK_ID --n_models ${n_models}

# Tidy the SLURM logs into this task's output folder (created by the run).
mkdir -p array_task${SLURM_ARRAY_TASK_ID}
mv slurm.run_search_${SLURM_ARRAY_TASK_ID}.out array_task${SLURM_ARRAY_TASK_ID}/ 2>/dev/null || true
mv slurm.run_search_${SLURM_ARRAY_TASK_ID}.err array_task${SLURM_ARRAY_TASK_ID}/ 2>/dev/null || true
