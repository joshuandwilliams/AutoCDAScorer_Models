#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 1
#SBATCH -p jic-gpu
#SBATCH --gres=gpu:1
#SBATCH --mem 4000
#SBATCH --time=0-00:10:00
#SBATCH --job-name="tf221_gpu_test"
#SBATCH -o test_tensorflow.out
#SBATCH -e test_tensorflow.err
#
# Smoke test for TensorFlowGPU_2_21_0.img: confirms TensorFlow is installed and
# can see + use a GPU on the jic-gpu queue.
#
# Requires the built image on SHARED storage that the compute nodes can read,
# i.e. under your home -- NOT in /tmp on the software node (that is node-local
# and invisible to jic-gpu nodes). Default location matches 01_01.slurm.sh;
# override with IMG=/path/to/the.img:
#   sbatch test_tensorflow.slurm.sh

img="${IMG:-$HOME/singularity/TensorFlowGPU_2_21_0/TensorFlowGPU_2_21_0.img}"

# 1. Show the GPU(s) the driver exposes on the allocated node.
singularity exec --nv "${img}" nvidia-smi

# 2. Report the installed TensorFlow version.
singularity exec --nv "${img}" pip show tensorflow

# 3. Confirm TensorFlow sees a GPU and can run an op on it. Exits non-zero (so
#    the SLURM job is marked FAILED) if no GPU is found.
singularity exec --nv "${img}" python3.12 -c "
import tensorflow as tf
print('TensorFlow:', tf.__version__)
print('Built with CUDA:', tf.test.is_built_with_cuda())
gpus = tf.config.list_physical_devices('GPU')
print('GPUs found:', len(gpus))
for g in gpus:
    print('  ', g)
if not gpus:
    raise SystemExit('FAIL: TensorFlow found no GPU')
with tf.device('/GPU:0'):
    a = tf.random.normal((2048, 2048))
    b = tf.matmul(a, a)
    _ = b.numpy()
print('GPU matmul OK, result shape:', b.shape)
print('PASS: TensorFlow GPU test succeeded')
"
