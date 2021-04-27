#!/bin/bash

#SBATCH --gres=gpu:1
#SBATCH --partition=n1s8-t4-1
#SBATCH --account=dl09
#SBATCH --time=23:00:00
#SBATCH --output=dl09_%j.out
#SBATCH --error=dl09_%j.err
#SBATCH --exclusive
#SBATCH --requeue
#SBATCH --job-name=dl09_pre
#SBATCH --mail-type=ALL
#SBATCH --mail-user=%u@nyu.edu

/share/apps/local/bin/p2pBandwidthLatencyTest > /dev/null 2>&1

set -x

mkdir /tmp/$USER
export SINGULARITY_CACHEDIR=/tmp/$USER

cp -rp /scratch/DL21SP/student_dataset.sqsh /tmp
echo "Dataset is copied to /tmp"

cd $HOME/dl09/src

singularity exec --nv \
--bind /scratch \
--overlay $SCRATCH/dl09/env/conda_lightly.ext3:ro \
--overlay /tmp/student_dataset.sqsh:ro \
/share/apps/images/cuda11.1-cudnn8-devel-ubuntu18.04.sif \
/bin/bash -c "
source /ext3/env.sh
conda activate dev
python pre_train.py --checkpoint-dir $SCRATCH/dl09/pre_checkpoints \
                    --save-checkpoint-per-epoch 5 \
                    --epochs 100 \
                    --batch-size 128 \
                    --workers 2 \
                    --gpus 1 \
                    --lr 0.006 \
                    --momentum 0.9 \
                    --weight-decay 5e-4 \
                    --moco-dim 128 \
                    --moco-k 4096 \
                    --moco-m 0.99 \
                    --moco-t 0.1 \
                    --num-splits 8 \
                    /dataset
"