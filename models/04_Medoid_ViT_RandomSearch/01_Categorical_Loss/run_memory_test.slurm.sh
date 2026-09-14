#!/bin/bash
#SBATCH -p jic-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem 16000
#SBATCH --time=0-06:00:00
#SBATCH --job-name="vit_mem_test"
# Written to the cwd then moved into logs/ at the end: SLURM opens these before the
# script body runs, so a logs/ path here would fail while the directory does not exist.
#SBATCH -o slurm.mem_test.out
#SBATCH -e slurm.mem_test.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=jowillia@nbi.ac.uk

# Answers the question the previous probe could not: does memory accumulate across
# models trained back-to-back inside ONE process?
#
# models/02_.../run_single_test.slurm.sh ran each config in its own `singularity exec`,
# so process teardown freed everything between configs. The real search does the
# opposite -- one array task trains hundreds of configs in a single process via
# run_grid's while-loop. Commit 13fa770 records the CNN sweep being OOM-killed exactly
# that way (host RSS climbing until the 8 GB task died 62/150 through the grid).
#
# Phase A trains 9 configs in one process that share an architecture and differ only in
# learning rate and dropout, so the workload is constant and any climb is a leak.
# Phase B trains the largest config the new grid can produce, to size --mem and --time.
#
# --mem is deliberately 16 GB, NOT the 40 GB the old search asked for. Over-provisioning
# would hide a slow leak by never hitting the ceiling; 16 GB is ample headroom over the
# 4.68 GB peak RSS measured in the single-config run while still leaving a real leak
# visible in the numbers before the job dies.
#
# --nv exposes the GPU to the container. Without it TensorFlow silently falls back to
# CPU, which is what happened to every ViT run in from_HPC/models_new (their logs show
# 'libcuda ... NOT_FOUND' while holding a GPU allocation). run_memory_test.py hard-fails
# if it cannot see a GPU.

set -uo pipefail

img="$HOME/singularity/TensorFlow/TensorFlowGPU_2_21_0.img"
mkdir -p logs

echo "=== nvidia-smi ==="
nvidia-smi || echo "WARNING: nvidia-smi unavailable on this node"

rc=0

echo ""
echo "=================================================================="
echo "=== PHASE A: 9 configs in ONE process (leak detection) ==="
echo "=================================================================="
singularity exec --nv "${img}" python3.12 run_memory_test.py --phase a
status=$?
[ ${status} -ne 0 ] && { echo "!!! phase A FAILED (exit ${status})"; rc=${status}; }

echo ""
echo "=================================================================="
echo "=== PHASE B: largest config in the new grid (headroom) ==="
echo "=================================================================="
singularity exec --nv "${img}" python3.12 run_memory_test.py --phase b
status=$?
[ ${status} -ne 0 ] && { echo "!!! phase B FAILED (exit ${status})"; rc=${status}; }

echo ""
echo "=================================================================="
echo "=== REPORT ==="
echo "=================================================================="
singularity exec --nv "${img}" python3.12 run_memory_test.py --report

mv slurm.mem_test.out logs/ 2>/dev/null || true
mv slurm.mem_test.err logs/ 2>/dev/null || true

exit ${rc}
