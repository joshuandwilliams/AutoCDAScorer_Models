#!/bin/bash
# One-command timing probe: trains ONE representative ViT config on a GPU and reports
# how long it took, so you can estimate the whole-grid wall clock before launching all
# 1,944 configs.
#   ./run_benchmark.sh
# When it finishes:  cat bench_result_*.csv   (seconds for a fixed 20-epoch fit)
# Rough total estimate ~= (that fit's seconds / 20) x 50 epochs x 5 folds x 1,944 configs
#   / (number of GPUs your --array actually gets).
# Uses the same jic-gpu / --gres / --nv recipe as run_search.slurm.sh.
img="$HOME/singularity/TensorFlow/TensorFlowGPU_2_21_0.img"
mkdir -p logs

sbatch -p jic-gpu --gres=gpu:1 --mem 16000 --time=0-01:00:00 --cpus-per-task=4 \
    --job-name="vit_bench_gpu" -o "logs/bench_gpu.out" -e "logs/bench_gpu.err" \
    --wrap "singularity exec --nv ${img} python3.12 bench_cpus.py"

echo "Submitted 1 GPU benchmark job."
echo "When it finishes:  cat bench_result_*.csv"
