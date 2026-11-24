#!/bin/bash
#SBATCH -p jic-compute
#SBATCH --cpus-per-task=4
#SBATCH --mem 48000
#SBATCH --time=2-00:00:00
#SBATCH --job-name="cnn_ganaug_match_tv"
#SBATCH -o slurm.run_search_%a.out
#SBATCH -e slurm.run_search_%a.err
#SBATCH --array=1-200
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=jowillia@nbi.ac.uk

# The 01_MatchLargestClass run with its training crops drawn from the train and
# validation splits pooled together, as models/03, 04 and 05 cross-validate over.
# 01_MatchLargestClass drew from the train split alone (see RERUN.md). Everything else is
# the same: run_search.py is a byte-identical copy, and the settings below are 01's.
#
# Crops topped up with cGAN SAMPLES, from
#   data/datasets/medoid_64_cgan_trainval/medoid_64_cgan.npz
# built by analyses/01_generate_base_dataset/build_trainval_pools.py. Results are written
# to this folder, so 01_MatchLargestClass is not touched.
#
# Resubmit the same command until random_search_results.csv holds 8,640 rows, as for 01.
#
img="$HOME/singularity/TensorFlow/TensorFlowGPU_2_21_0.img"
top_n="${TOP_N:-100}"
n_configs="${N_CONFIGS:-0}"
max_configs="${MAX_CONFIGS:-44}"
max_rss_mb="${MAX_RSS_MB:-32000}"
per_class="${PER_CLASS:-0}"

echo "settings: per_class=${per_class} n_configs=${n_configs} max_configs=${max_configs} max_rss_mb=${max_rss_mb}"

singularity exec ${img} python3.12 run_search.py \
    --slurm_array $SLURM_ARRAY_TASK_ID --per_class ${per_class} \
    --n_configs ${n_configs} --top_n ${top_n} \
    --max_configs ${max_configs} --max_rss_mb ${max_rss_mb} \
    --pool ../../../data/datasets/medoid_64_cgan_trainval/medoid_64_cgan.npz
rc=$?
[ ${rc} -ne 0 ] && echo "!!! run_search.py FAILED with exit code ${rc}"

mkdir -p logs
mv slurm.run_search_${SLURM_ARRAY_TASK_ID}.out logs/ 2>/dev/null || true
mv slurm.run_search_${SLURM_ARRAY_TASK_ID}.err logs/ 2>/dev/null || true

exit ${rc}
