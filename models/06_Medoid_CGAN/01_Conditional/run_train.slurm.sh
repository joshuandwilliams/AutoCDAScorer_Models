#!/bin/bash
#SBATCH -p jic-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem 24000
#SBATCH --time=0-04:00:00
#SBATCH --job-name="medoid_cgan"
#SBATCH -o slurm.cgan_%a.out
#SBATCH -e slurm.cgan_%a.err
#SBATCH --array=1-64
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=jowillia@nbi.ac.uk

# Conditional GAN over a 64-config grid, trained on the medoid_64 TRAIN split only.
#
# One config per array task, NOT self-scheduling. GAN training is one long adversarial run
# rather than many short CV fits, so there is nothing for the JobCounter to schedule and no
# per-config memory budget to enforce -- each task builds two models once and trains them.
#
# --nv is required. Without it the container cannot see the GPU and TensorFlow silently
# falls back to CPU while still holding the GPU allocation.
#
# COST IS BOUNDED BY --max_hours, NOT BY THE EPOCH COUNT. The Frechet early-stopping rule
# only fires while the discriminator's accuracy is inside [0.35, 0.65], so a config whose
# discriminator has won outright never triggers it and would otherwise run all 5,000
# epochs. At 3 h per config the grid cannot exceed 192 GPU-hours however badly the configs
# behave, and jic-gpu has 26 A100s. The SLURM --time of 4 h is slack above that budget so
# the Python side always exits first and still writes its results CSV.
#
# --array=1-64, no "%" throttle: 64 is the exact grid size and the account limit is 200
# concurrent jobs, so the partition's GPU count is the only thing throttling this.

img="$HOME/singularity/TensorFlow/TensorFlowGPU_2_21_0.img"
epochs="${EPOCHS:-5000}"
max_hours="${MAX_HOURS:-3}"

echo "settings: epochs=${epochs} max_hours=${max_hours} config=${SLURM_ARRAY_TASK_ID}/64"
nvidia-smi || echo "WARNING: nvidia-smi unavailable"

singularity exec --nv ${img} python3.12 run_train.py \
    --slurm_array $SLURM_ARRAY_TASK_ID --epochs ${epochs} --max_hours ${max_hours}
rc=$?
# Capture and re-raise the Python exit code. Without this the trailing mv commands set
# the script's status and SLURM records COMPLETED 0:0 even when training crashed --
# which is exactly what happened on the first smoke test (a scipy TypeError produced a
# "successful" job with no output files at all).
if [ ${rc} -ne 0 ]; then
    echo "!!! run_train.py FAILED with exit code ${rc}"
fi

mkdir -p logs
mv slurm.cgan_${SLURM_ARRAY_TASK_ID}.out logs/ 2>/dev/null || true
mv slurm.cgan_${SLURM_ARRAY_TASK_ID}.err logs/ 2>/dev/null || true

exit ${rc}
