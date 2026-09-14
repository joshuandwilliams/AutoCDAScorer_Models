#!/bin/bash
#SBATCH -p jic-compute
#SBATCH --cpus-per-task=4
#SBATCH --mem 32000
#SBATCH --time=1-00:00:00
#SBATCH --job-name="medoid_cnn_soft"
#SBATCH -o slurm.run_search_%a.out
#SBATCH -e slurm.run_search_%a.err
#SBATCH --array=1-240
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=jowillia@nbi.ac.uk

# Medoid CNN grid search trained on SOFT labels: the distribution of the three raters'
# scores instead of a one-hot of their median. Everything else -- dataset, splits,
# PARAM_SPACE, k=5, cv_seed=42 -- is identical to models/03_Medoid_CNN_RandomSearch, so
# the pair is a controlled comparison of the training target alone.
#
# CPU queue, as models/03 used: these CNNs are tiny and the GPU queue pends far longer
# than the speed-up is worth. The container is GPU-built but runs fine on CPU (TensorFlow
# logs a harmless "no GPU found"), hence no --nv and no --gres.
#
# Resource settings differ from models/03 in two ways, both learned from the ViT run:
#
#   --max_configs 40 and --max_rss_mb 24000. TensorFlow retains memory across repeated
#     model.fit calls and neither del, clear_session() nor gc.collect() releases it
#     (tensorflow/tensorflow#113529, against the TF 2.21.0 this container pins). models/03
#     ran before that was understood; commit 13fa770 records the earlier Base-CNN sweep
#     being OOM-killed 62/150 through the grid for exactly this reason. A task now exits
#     CLEANLY after 40 configs and a queued task resumes from the shared counter. Exiting
#     matters rather than merely surviving: JobCounter.claim() advances next.txt BEFORE
#     training, so a task killed mid-config leaves that index claimed but never trained,
#     absent from the results with no error raised anywhere.
#
#   --mem 32000 (models/03 used 8000). Headroom for 40 configs of leak; the RSS ceiling
#     above stops the task before SLURM does.
#
#   --array=1-240. 8,640 configs / 40 per task = 216 task-lifetimes, plus spare.
#     Surplus tasks find the grid exhausted, delete their empty part file and exit.

img="$HOME/singularity/TensorFlow/TensorFlowGPU_2_21_0.img"
top_n="${TOP_N:-100}"
max_configs="${MAX_CONFIGS:-40}"
max_rss_mb="${MAX_RSS_MB:-24000}"

echo "settings: top_n=${top_n} max_configs=${max_configs} max_rss_mb=${max_rss_mb}"

singularity exec ${img} python3.12 run_search.py \
    --slurm_array $SLURM_ARRAY_TASK_ID --top_n ${top_n} \
    --max_configs ${max_configs} --max_rss_mb ${max_rss_mb}

# Move this task's SLURM logs into the shared logs/ folder.
mkdir -p logs
mv slurm.run_search_${SLURM_ARRAY_TASK_ID}.out logs/ 2>/dev/null || true
mv slurm.run_search_${SLURM_ARRAY_TASK_ID}.err logs/ 2>/dev/null || true
