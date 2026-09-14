#!/bin/bash
#SBATCH -p jic-compute
#SBATCH --cpus-per-task=4
#SBATCH --mem 72000
#SBATCH --time=4-00:00:00
#SBATCH --job-name="cnn_aug_5000"
#SBATCH -o slurm.run_search_%a.out
#SBATCH -e slurm.run_search_%a.err
#SBATCH --array=1-200
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=jowillia@nbi.ac.uk

# Medoid CNN on crops topped up with GEOMETRIC AUGMENTATIONS to 5,000 per class, so every
# class including score 0 is mostly augmented. Tests what happens past the point where
# augmentation is only correcting class imbalance.
#
# The FULL 8,640-config grid, byte-identical to models/03_Medoid_CNN_RandomSearch's
# PARAM_SPACE, so this run differs from models/03 in the training data alone and every
# config can be paired with its models/03 counterpart on the twelve parameter columns.
#
# 200 array tasks, which is the account's running-job limit, so one submission saturates
# it and four runs together stay well inside the 3,000-job submit limit.
#
# 8,640 configs over 200 tasks is 43.2 each, hence --max_configs 44: enough for a single
# submission to cover the whole grid. At about 1.58 h per config on this run's crops that
# is roughly 68 h per task, which is why --time is 4-00:00:00 rather than one day.
#
# In practice --max_rss_mb usually fires first. That is not a failure: the task exits
# cleanly, and RESUBMITTING THE SAME COMMAND resumes from the shared counter and trains
# only the configs still outstanding. Repeat until random_search_results.csv holds 8,640
# rows.
#
# --max_configs with --max_rss_mb: TensorFlow retains memory across repeated model.fit
# calls and clear_session does not release it (tensorflow/tensorflow#113529 against the
# TF 2.21.0 this container pins). A task exits CLEANLY after its budget and a queued task
# resumes from the shared counter. Exiting cleanly matters because JobCounter.claim()
# advances next.txt BEFORE training, so a task killed mid-config leaves that index claimed
# but never trained and silently absent from the results.
#
# To run a cheaper subset instead, set N_CONFIGS to restrict the search to that many of
# models/03's best configs (N_CONFIGS=100 costs about 55 and 168 CPU-hours respectively).
#
# Training crops are sampled from data/datasets/medoid_64_aug with subset_seed 7, shared
# between both runs, so 01's training set is a strict subset of 02's. Folds are grouped on
# the source crop, so an augmented copy is never split from the original it was made from.
# Test is the balanced 308-crop medoid_64 test split, identical to models/03 and models/05.
#
# CPU queue, as models/03 and models/05 used: these CNNs are tiny and the GPU queue pends
# longer than the speed-up is worth. The container is GPU-built but runs on CPU fine.
#
img="$HOME/singularity/TensorFlow/TensorFlowGPU_2_21_0.img"
top_n="${TOP_N:-100}"
n_configs="${N_CONFIGS:-0}"
max_configs="${MAX_CONFIGS:-44}"
max_rss_mb="${MAX_RSS_MB:-56000}"
per_class="${PER_CLASS:-5000}"

echo "settings: per_class=${per_class} n_configs=${n_configs} max_configs=${max_configs} max_rss_mb=${max_rss_mb}"

singularity exec ${img} python3.12 run_search.py \
    --slurm_array $SLURM_ARRAY_TASK_ID --per_class ${per_class} \
    --n_configs ${n_configs} --top_n ${top_n} \
    --max_configs ${max_configs} --max_rss_mb ${max_rss_mb}
rc=$?
[ ${rc} -ne 0 ] && echo "!!! run_search.py FAILED with exit code ${rc}"

mkdir -p logs
mv slurm.run_search_${SLURM_ARRAY_TASK_ID}.out logs/ 2>/dev/null || true
mv slurm.run_search_${SLURM_ARRAY_TASK_ID}.err logs/ 2>/dev/null || true

exit ${rc}
