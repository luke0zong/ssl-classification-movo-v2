import os
import argparse
import torch
import torch.nn as nn
import torchvision
import lightly
import pytorch_lightning as pl
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint

from dataloader import CustomDataset
from moco import MocoModel
from classifier import Classifier

parser = argparse.ArgumentParser(description='dl09 pretrain moco')

#### dataset path
parser.add_argument('data', metavar='DIR',
                    help='dir to dataset')
#### trainer params
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--save-checkpoint-per-epoch', default=5, type=int, metavar='N',
                    help='save checkpoint on how many opechs (default: 5)')
parser.add_argument('--checkpoint-dir', default='../checkpoints', type=str, metavar='DIR',
                    help='dir to save checkpoints')
parser.add_argument('--pretrain-dir', default='../checkpoints', type=str, metavar='DIR',
                    help='dir to pretrained checkpoints')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--gpus', default=1, type=int,
                    help='number of gpus to use')
#### optim params
parser.add_argument('--lr', '--learning-rate', default=0.0075, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum of SGD solver')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
#### moco configs
parser.add_argument('--moco-dim', default=128, type=int,
                    help='feature dimension (default: 128)')
parser.add_argument('--moco-k', default=4096, type=int,
                    help='queue size; number of negative keys (default: 4096)')
parser.add_argument('--moco-m', default=0.99, type=float,
                    help='moco momentum of updating key encoder (default: 0.99)')
parser.add_argument('--moco-t', default=0.2, type=float,
                    help='softmax temperature (default: 0.2)')
parser.add_argument('--num-splits', default=8, type=int,
                    help='Split number for SplitBatchNorm (default:8)')


def main():
    args = parser.parse_args()

    if not os.path.isdir(args.pretrain_dir]):
        raise ValueError(f"pretrain dir does not exist: {args.pretrain_dir]}")

    if args.checkpoint_dir:
        if not os.path.isdir(args.checkpoint_dir):
            raise ValueError(f"checkpoint dir does not exist: {args.checkpoint_dir}")

    ###################### training starts! ######################    
    #### load model
    print("=> loading pretrained model ")
    moco = MocoModel(moco_dim=args.moco_dim,
                      moco_k=args.moco_k,
                      moco_m=args.moco_m,
                      moco_t=args.moco_t,
                      num_splits=args.num_splits,
                      lr=args.lr,
                      momentum=args.momentum,
                      weight_decay=args.weight_decay)
    moco.load_from_checkpoint(args.pretrain_dir)
    moco.eval()

    #### create cls
    print("=> creating classifier model")
    model = Classifier(moco)
    del moco

    #### create augmentation
    train_classifier_transforms = torchvision.transforms.Compose([
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(
        mean=lightly.data.collate.imagenet_normalize['mean'],
        std=lightly.data.collate.imagenet_normalize['std'])])
    # No additional augmentations for the test set
    test_transforms = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(
        mean=lightly.data.collate.imagenet_normalize['mean'],
        std=lightly.data.collate.imagenet_normalize['std'])])

    train_dataset = CustomDataset(args.data, 'train', train_classifier_transforms)
    eval_dataset = CustomDataset(args.data, 'val', test_transforms)

    dataset_train_classifier = lightly.data.LightlyDataset.from_torch_dataset(train_dataset)
    dataset_eval_classifier = lightly.data.LightlyDataset.from_torch_dataset(eval_dataset)

    dataloader_train_classifier = torch.utils.data.DataLoader(
        dataset_train_classifier,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.workers)

    dataloader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.workers)

    #### create trainer
    checkpoint_callback = ModelCheckpoint(dirpath=args.checkpoint_dir,
                                          filename='pretrain_{epoch:03d}_{loss:.2f}',
                                          period=args.save_checkpoint_per_epoch)
    trainer = pl.Trainer(max_epochs=args.epochs, 
                         gpus=args.gpus,
                         progress_bar_refresh_rate=100,
                         overfit_batches=overfit_batches,
                         resume_from_checkpoint=checkpoint_path,
                         benchmark=True,
                         callbacks=[checkpoint_callback]) 

    print("=> Start training")
    trainer.fit(model, dataloader_train_classifier, dataloader_test)


if __name__ == '__main__':
    main()
