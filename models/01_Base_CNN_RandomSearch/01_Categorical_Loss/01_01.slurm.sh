#!/bin/bash
#SBATCH -p jic-gpu
#SBATCH --mem 64000
#SBATCH --job-name="base_cnn_01_01"
#SBATCH -o slurm.01_01_%a.out
#SBATCH -e slurm.01_01_%a.err
#SBATCH --array=1-45
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=jowillia@nbi.ac.uk


img="$HOME/singularity/TensorFlow/TensorFlowGPU_2_21_0.img"
per_array=6

singularity exec --nv ${img} python3.12 01_01.py --slurm_array $SLURM_ARRAY_TASK_ID --per_array ${per_array}

# Tidy the SLURM logs into this task's output folder (created by train_model).
mkdir -p array_task${SLURM_ARRAY_TASK_ID}
mv slurm.01_01_${SLURM_ARRAY_TASK_ID}.out array_task${SLURM_ARRAY_TASK_ID}/ 2>/dev/null || true
mv slurm.01_01_${SLURM_ARRAY_TASK_ID}.err array_task${SLURM_ARRAY_TASK_ID}/ 2>/dev/null || true
