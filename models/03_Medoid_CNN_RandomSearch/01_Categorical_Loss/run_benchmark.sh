#!/bin/bash
# One-command CPU-scaling benchmark. Submits the SAME model at 2, 4 and 8 CPUs.
#   ./run_benchmark.sh
# When the 3 jobs finish:  cat bench_result_*.csv   (compare seconds vs cpus)
img="$HOME/singularity/TensorFlow/TensorFlowGPU_2_21_0.img"
mkdir -p logs

for n in 2 4 8; do
    sbatch -p jic-compute --mem 8000 --time=0-00:30:00 --cpus-per-task=${n} \
        --job-name="cpu_bench_${n}" -o "logs/bench_${n}.out" -e "logs/bench_${n}.err" \
        --wrap "singularity exec ${img} python3.12 bench_cpus.py"
done

echo "Submitted 3 benchmark jobs (2, 4, 8 CPUs)."
echo "When they finish:  cat bench_result_*.csv"
