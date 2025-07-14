#!/bin/bash
#SBATCH -p jic-short
#SBATCH --mem 2000
#SBATCH --job-name="cnn_find_and_concatenate_results"
#SBATCH -o slurm.cnn_find_and_concatenate_results.out
#SBATCH -e slurm.cnn_find_and_concatenate_results.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=jowillia@nbi.ac.uk

img="$HOME/singularity/TensorFlowGPU_2_16_1/TensorFlowGPU_2_16_1.img"
singularity exec ${img} python3.10 combine.py
