#!/bin/bash
#SBATCH -p jic-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem 16000
#SBATCH --time=1-00:00:00
#SBATCH --job-name="base_vit_search"
#SBATCH -o slurm.run_search_%a.out
#SBATCH -e slurm.run_search_%a.err
#SBATCH --array=1-16
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=jowillia@nbi.ac.uk

# Same self-scheduling FULL GRID SEARCH strategy as the Base CNN sweep, with the ViT
# builder and its broad 1,944-config grid. Each worker task pulls the next unclaimed
# config from a shared counter (next.txt) until the whole grid is done, all contributing
# to ONE shared global top-100 (results/, threshold.txt, random_search_results.csv).
# Load-balances automatically; a re-submit resumes from the counter.
#
# On the GPU queue (jic-gpu, --gres=gpu:1, --nv), unlike the CNN sweep: a ViT is far
# heavier than the tiny CNNs (attention over up to 256 patches, 5-fold CV x up to 50
# epochs), and this grid is broad, so the GPU speed-up is worth the queue pend. The
# jic-gpu / --gres / --nv recipe matches containers/test_tensorflow.slurm.sh, which is
# the verified way to run this exact image on a GPU. --nv exposes the GPU to the
# container; without it TensorFlow silently falls back to CPU.
#
# --array=1-16 asks for up to 16 concurrent GPUs; SLURM runs as many as your QOS allows
# and queues the rest -- because the tasks self-schedule, the exact number is not
# critical (more GPUs = shorter wall clock). 4 CPUs is plenty to feed the in-memory
# dataset. --time is 1 day per task; a re-submit resumes from the shared counter.
img="$HOME/singularity/TensorFlow/TensorFlowGPU_2_21_0.img"
top_n=100

singularity exec --nv ${img} python3.12 run_search.py --slurm_array $SLURM_ARRAY_TASK_ID --top_n ${top_n}

# Move this task's SLURM logs into the shared logs/ folder.
mkdir -p logs
mv slurm.run_search_${SLURM_ARRAY_TASK_ID}.out logs/ 2>/dev/null || true
mv slurm.run_search_${SLURM_ARRAY_TASK_ID}.err logs/ 2>/dev/null || true
