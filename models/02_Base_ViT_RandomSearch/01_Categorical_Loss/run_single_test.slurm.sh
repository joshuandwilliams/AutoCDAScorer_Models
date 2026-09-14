#!/bin/bash
#SBATCH -p jic-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem 40000
#SBATCH --time=0-12:00:00
#SBATCH --job-name="vit_single_test"
# Written to the cwd, then moved into logs/ at the end. SLURM opens these before the
# script body runs, so a logs/ path here would fail when the directory does not yet
# exist. Same pattern as run_search.slurm.sh.
#SBATCH -o slurm.single_test.out
#SBATCH -e slurm.single_test.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=jowillia@nbi.ac.uk

# ONE GPU job that trains four pinned ViT configs through the real search harness,
# before committing to the 1,944-config grid. It answers two questions:
#
#   1. Does the grid code path work on a GPU? The full grid has never produced a result.
#      An earlier attempt ran out of GPU memory, and the three fixes for that (per-fold
#      graph teardown in src/random_search.py, batch 64 -> 32, more --mem) have not been
#      exercised on a device. The 'worst' config below is the one that OOM'd.
#
#   2. What does the grid cost? p4/p8/p16 vary only patch_size, which sets the token
#      count (256/64/16) that attention is quadratic in. Exactly 648 of the 1,944 grid
#      configs use each patch size, so those three time a representative config in each
#      third of the grid.
#
# Resources deliberately MATCH run_search.slurm.sh (jic-gpu, 1 GPU, 4 CPUs, 40 GB) so
# the memory result transfers directly to the real run. --time is 12 h, which is far
# more than four configs should need; it is slack, not an estimate.
#
# --nv is what exposes the GPU to the container. Without it TensorFlow silently falls
# back to CPU -- which is what happened to every ViT run in from_HPC/models_new, whose
# logs show 'libcuda ... NOT_FOUND' despite holding a GPU allocation. run_single_test.py
# hard-fails if it cannot see a GPU, so that failure mode cannot pass unnoticed here.

set -uo pipefail

img="$HOME/singularity/TensorFlow/TensorFlowGPU_2_21_0.img"
mkdir -p logs

echo "=== nvidia-smi ==="
nvidia-smi || echo "WARNING: nvidia-smi unavailable on this node"

rc=0
for cfg in p16 p8 p4 worst; do
    echo ""
    echo "=================================================================="
    echo "=== config: ${cfg}   ($(date '+%Y-%m-%d %H:%M:%S')) ==="
    echo "=================================================================="
    singularity exec --nv "${img}" python3.12 run_single_test.py --config "${cfg}"
    status=$?
    if [ ${status} -ne 0 ]; then
        echo "!!! config ${cfg} FAILED with exit code ${status}"
        rc=${status}
        # Keep going: a large config failing is itself a result, and the smaller ones
        # still give a usable cost model.
    fi
done

echo ""
echo "=================================================================="
echo "=== summary ==="
echo "=================================================================="
singularity exec --nv "${img}" python3.12 run_single_test.py --summarise

# Move this job's SLURM logs into logs/, as run_search.slurm.sh does.
mv slurm.single_test.out logs/ 2>/dev/null || true
mv slurm.single_test.err logs/ 2>/dev/null || true

exit ${rc}
