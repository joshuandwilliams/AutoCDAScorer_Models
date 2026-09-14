#!/bin/bash
#SBATCH -p jic-short
#SBATCH --cpus-per-task=4
#SBATCH --mem 16000
#SBATCH --time=0-00:30:00
#SBATCH --job-name="gan_diversity_features"
#SBATCH -o slurm.features.out
#SBATCH -e slurm.features.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=jowillia@nbi.ac.uk

# Feature extraction for the GAN diversity diagnostic. Batch job, not the login node:
# this reads ~1,834 real crops plus 1,834 sampled generated ones off a shared parallel
# filesystem and runs them all through a CNN, which is far past what the login node
# should be asked to do.
#
# CPU queue and no --nv: this is a few thousand forward passes through a 88k-parameter
# CNN, so the GPU queue's pend time would exceed the whole job. The container is
# GPU-built but runs on CPU fine (TensorFlow logs a harmless "no GPU found").
#
# Writes analyses/07_gan_diversity/features.npz, which is small (a few MB of feature
# vectors). Pull it down with sync_from_hpc and do the analysis locally.

set -uo pipefail

img="$HOME/singularity/TensorFlow/TensorFlowGPU_2_21_0.img"
gan_dir="${GAN_DIR:-$HOME/data/GAN_images_2}"

echo "extracting from ${gan_dir}"
singularity exec ${img} python3.12 extract_features.py --gan_dir "${gan_dir}"
rc=$?
[ ${rc} -ne 0 ] && echo "!!! extract_features.py FAILED with exit code ${rc}"

mkdir -p logs
mv slurm.features.out logs/ 2>/dev/null || true
mv slurm.features.err logs/ 2>/dev/null || true

exit ${rc}
