#!/bin/bash
#SBATCH -p jic-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem 16000
#SBATCH --time=0-02:00:00
#SBATCH --job-name="base_cnn_01_01"
#SBATCH -o slurm.01_01_%a.out
#SBATCH -e slurm.01_01_%a.err
#SBATCH --array=1-1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=jowillia@nbi.ac.uk

# Small check run: one array task training 4 randomly sampled models. To scale up,
# raise --array (each task uses its index as the RNG seed) and/or --n_models.
img="$HOME/singularity/TensorFlow/TensorFlowGPU_2_21_0.img"
n_models=4

singularity exec --nv ${img} python3.12 01_01.py --slurm_array $SLURM_ARRAY_TASK_ID --n_models ${n_models}

# Tidy the SLURM logs into this task's output folder (created by the run).
mkdir -p array_task${SLURM_ARRAY_TASK_ID}
mv slurm.01_01_${SLURM_ARRAY_TASK_ID}.out array_task${SLURM_ARRAY_TASK_ID}/ 2>/dev/null || true
mv slurm.01_01_${SLURM_ARRAY_TASK_ID}.err array_task${SLURM_ARRAY_TASK_ID}/ 2>/dev/null || true
