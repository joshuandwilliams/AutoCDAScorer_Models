#!/bin/bash
#SBATCH -p jic-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem 96000
#SBATCH --time=2-00:00:00
#SBATCH --job-name="medoid_vit_search"
#SBATCH -o slurm.run_search_%a.out
#SBATCH -e slurm.run_search_%a.err
#SBATCH --array=1-240
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=jowillia@nbi.ac.uk

# Self-scheduling FULL GRID SEARCH over the 2,592-config ViT grid on the medoid dataset.
# Each worker pulls the next unclaimed config from a shared counter (next.txt) until the
# grid is done, all contributing to ONE shared global top-100 (results/, threshold.txt,
# random_search_results.csv). Load-balances automatically; a re-submit resumes from the
# counter. Tasks that start after the grid is exhausted exit immediately and clean up
# their own empty part file, so an oversized array costs nothing.
#
# THE MEMORY TEST FAILED, AND THESE SETTINGS ARE THE RESPONSE TO IT (job 23171521,
# OUT_OF_MEMORY, MaxRSS 15.45 GB against a 15.62 GB request). Phase A trained nine
# configs of ONE fixed architecture in a single process, so the workload was constant,
# and host RSS still climbed monotonically:
#
#     config 0  4,627 MB      config 3  11,904 MB
#     config 1  6,770 MB      config 4  14,556 MB
#     config 2  8,899 MB      config 5  killed
#
# That is ~2.5 GB retained per config. It is not this project's bug. TensorFlow retains
# memory across repeated model.fit calls and neither del, clear_session() nor
# gc.collect() releases it -- tensorflow/tensorflow#113529 reports exactly this against
# TF 2.21.0, the version containers/TensorFlowGPU_2_21_0.def pins. I reproduced it
# locally and bisected it: building and deleting models in a loop is completely flat,
# and the growth appears only once model.fit is called. GPU memory leaks too, at a
# steady +51 MB per config, but slowly enough not to matter.
#
# Resource settings, and why they are what they are:
#
#   --max_configs 20 (passed below). THE KEY SETTING. Each task trains 20 configs then
#     EXITS CLEANLY, and a queued array task resumes from the shared counter. Exiting
#     matters rather than merely surviving: JobCounter.claim() advances next.txt BEFORE
#     training, so a task killed mid-config leaves that index claimed but never trained,
#     absent from the results with no error raised anywhere. A clean exit runs
#     _merge_part and loses nothing. At the measured 2.5 GB/config, 20 configs is ~53 GB;
#     at a pessimistic 4 GB/config it is ~83 GB. Both sit inside the 96 GB below.
#
#   --mem 96000. Every jic-gpu node has 1,025,000 MB of RAM and at most 4 GPUs, so four
#     96 GB tasks use 384 GB of 1 TB and a node can still fill all its GPUs. Memory does
#     NOT limit GPU concurrency on this partition -- GPUs do.
#
#   --array=1-240, no "%" throttle, so nothing here caps how many run at once. A
#     20-config budget needs ceil(2592/20) = 130 task-lifetimes, but the leak scales with
#     model size, so the largest configs will trip the RSS ceiling before the budget: the
#     2-task smoke run measured ~0.9 GB per config on the grid's small early configs,
#     against 2.5 GB per config for proj 128 / depth 6 in the memory test. At a
#     pessimistic 5 GB the 80 GB ceiling allows only ~16 configs per task, needing 162
#     lifetimes. 240 covers that case with room for node failures. Surplus tasks cost
#     nothing -- they find the grid exhausted, delete their empty part file and exit.
#     At most 26 run simultaneously
#     regardless -- that is every A100 on jic-gpu (4+1+1+2+2+2+2+4+4+4 across
#     j1024n2-n11) -- so the rest are a queue that feeds itself as tasks retire. Account
#     limits are well clear: MaxJobs 200, MaxSubmit 3000, MaxArraySize 6000.
#
#     Note the tail is NOT a concern here: work is pulled dynamically from the shared
#     counter, so no task can be stuck with an unlucky static slice, and the tail at the
#     end of the run is bounded by one config's runtime.
#
#   --time 2 days per task. The probe measured 340-512 s per config at 5-fold CV for
#     the old grid's sizes. The new grid reaches larger models but runs at batch 64
#     (fewer steps per epoch), so per-config cost is not yet pinned down -- phase B of
#     the memory test measures it. Two days is deliberately generous, and a re-submit
#     resumes from the counter regardless.
#
# --nv exposes the GPU to the container. Without it TensorFlow silently falls back to
# CPU -- the failure mode that quietly degraded every ViT run in from_HPC/models_new.
# The jic-gpu / --gres / --nv recipe matches containers/test_tensorflow.slurm.sh.

img="$HOME/singularity/TensorFlow/TensorFlowGPU_2_21_0.img"
# Overridable from the environment so a smoke test can use different values without
# editing this file, e.g.
#   sbatch --array=1-2 --export=ALL,MAX_CONFIGS=3 run_search.slurm.sh
top_n="${TOP_N:-100}"
max_configs="${MAX_CONFIGS:-20}"
max_rss_mb="${MAX_RSS_MB:-80000}"  # below --mem 96000, so the task ends itself first

echo "settings: top_n=${top_n} max_configs=${max_configs} max_rss_mb=${max_rss_mb}"

singularity exec --nv ${img} python3.12 run_search.py \
    --slurm_array $SLURM_ARRAY_TASK_ID --top_n ${top_n} \
    --max_configs ${max_configs} --max_rss_mb ${max_rss_mb}

# Move this task's SLURM logs into the shared logs/ folder.
mkdir -p logs
mv slurm.run_search_${SLURM_ARRAY_TASK_ID}.out logs/ 2>/dev/null || true
mv slurm.run_search_${SLURM_ARRAY_TASK_ID}.err logs/ 2>/dev/null || true
