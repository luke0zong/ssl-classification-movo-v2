# Self-Supervised Learning Image Cllassification with Momentum Contrast



> This is the final competition for NYU course CSCI-GA 2572 Deep Learning Spring 2021.
>
> Instructor: Yann LeCun and Alfredo Canziani.


Deep Learning Final Competition Team DL09

**Members:**

Arthur Jinyue Guo (jg5505)

Yaowei Zong (yz7413)


---

## Code Structure

### src

source code

- model
  - Adopted MoCO v2
    - https://github.com/facebookresearch/moco
    - https://medium.com/analytics-vidhya/simclr-with-less-computational-constraints-moco-v2-in-pytorch-3d8f3a8f8bf2

### scripts

Sbatch files to submit jobs on Greene HPC

- pretrain.sbatch:

  - pretraining on unlabeled images for 100 epochs.

- pretrain_continue.sbatch:

  - script for resume pretraining.

- train.sbatch:

  - training on train dataset for 100 epochs.

- train_extra.sbatch:

  - training with extra labels.

- finetune.sbatch

  - run fine tuning after training.

- eval.sbatch:
  - evaluating on validation set.

## Replicate Instructions

(change dl09 to your hpc account for sbatch scipts and directories)

1. Request a CPU node on Greene log-4.
2. Clone [this](https://github.com/luke0zong/dl09.git) repo on the CPU node (GCP node).
3. Make sure you have the following files and directories:

   ```bash
   # data file
   /scratch/DL21SP/student_dataset.sqsh

   # env files
   /scratch/DL21SP/conda_fixed.sqsh
   /share/apps/images/cuda11.1-cudnn8-devel-ubuntu18.04.sif

   # log dir
   $SCRATCH/log/

   # pretrain checkpoints dir
   $SCRATCH/dl09/pre_checkpoints

   # train checkpoints dir
   $SCRATCH/dl09/checkpoints

   # train with extra label checkpoints dir
   $SCRATCH/dl09/extra_checkpoints

   # finetune dir (optional if you don't run fineturn)
   $SCRATCH/dl09/fine_tune_checkpoints
   ```

4. Pretraining:
   run the following (takes > 60 hours with single T4 GPU)

   ```bash
   cd $HOME/dl09/scripts
   sbatch pretrain.sbatch
   ```

   Note: if the pretraining job got killed, change the following args to the correct checkpoint and run `sbatch pretrain_continue.sbatch` to continue the training.

   ```bash
   --resume $SCRATCH/dl09/pre_checkpoints/checkpoint_082.pth
   --start-epoch 83
   ```

5. Training:
   run the following (takes > 1 hour with single T4 GPU)

   ```bash
   cd $HOME/dl09/scripts
   sbatch train.sbatch
   ```

   **Note:** if you didn't finish 100 epochs, change the `--pretrained $SCRATCH/dl09/pre_checkpoints/checkpoint_100.pth` inside `train.sbatch` to the correct checkpoint file.

6. Training with extra labels:

   Prepare the extra train image dataset with corresponding labeling file.

   Change the line inside `train_extra.sbatch` with your training dataset:

   ```dash
   cp -rp /scratch/jg5505/dl09/dataset /tmp/dataset
   ```


   Then run

   ```bash
   sbatch train_extra.sbatch
   ```

7. Fine Tuning (optional):

   If you run finetuning, make sure to change `finetune.sbatch` with the correct checkpoint file from the step above: `model_sub.pth`. (`model_best_{epoch}_{acc}.pth` contains extra states)

   You might get worse result. We didn't use the result from this for submission.

8. Evaluating:

   To test the model on validation dataset, put the checkpoint(`model_sub.pth`) from training step in this location: `$SCRATCH/model.pth`, then run

   ```
   sbatch eval.sbatch
   ```

   Make sure you have `src/submission.py` correctly defined.

### Labeling request

At the time of labeling request task, our model was not performing as expected.

Our initial plan for selecting bad images is:


- During the pre-training stage, a random sample of size 12,800 images from the unlabeled dataset is selected.

- After pre-training for certain epochs, the indices of such sampler dataset and the losses and accuracy are recorded.

- Repeat the above step 40 times (i.e 512,000/12,800 such random samplers), the images with the worst losses and accuracies are selected for this job.

We ended up selecting 12,800 random indices from the unlabeled dataset.


## Result

Due to time limiting and lacking one team member ( other teams have 3 or 4 members), we didn't have enough time to run a full pretraining and finetuning, thus the final submission only has a **15.98%** validation accuracy (5.94% with extra labels).
