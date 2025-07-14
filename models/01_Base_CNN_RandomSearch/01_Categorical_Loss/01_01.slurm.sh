#!/bin/bash
#SBATCH -p jic-gpu
#SBATCH --mem 64000
#SBATCH --job-name="CGAN_CNN_1"
#SBATCH -o slurm.CGAN_CNN_1_%a.out
#SBATCH -e slurm.CGAN_CNN_1_%a.err
#SBATCH --array=1-45
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=jowillia@nbi.ac.uk


img="$HOME/singularity/TensorFlowGPU_2_16_1/TensorFlowGPU_2_16_1.img"
per_array=6

singularity exec ${img} python3.10 CGAN_CNN_1.py --slurm_array $SLURM_ARRAY_TASK_ID --per_array ${per_array}

mv slurm.base_CNN_${SLURM_ARRAY_TASK_ID}.err array_task${SLURM_ARRAY_TASK_ID}/
mv slurm.base_CNN_${SLURM_ARRAY_TASK_ID}.out array_task${SLURM_ARRAY_TASK_ID}/
