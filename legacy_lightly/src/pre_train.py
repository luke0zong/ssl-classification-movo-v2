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
#### test run config
# test train option
parser.add_argument('--small-set', action='store_true',
                    help='use smaller training set (1/10)')


def main():
    args = parser.parse_args()

    if args.checkpoint_dir:
        if not os.path.isdir(args.checkpoint_dir):
            raise ValueError(f"checkpoint dir does not exist: {args.checkpoint_dir}")

    if args.resume:
        if not os.path.isfile(args.resume):
            raise ValueError(f"resume checkpoint does not exist: {args.resume}")

    ###################### training starts! ######################    
    #### load model
    print("=> creating model ")
    model = MocoModel(moco_dim=args.moco_dim,
                      moco_k=args.moco_k,
                      moco_m=args.moco_m,
                      moco_t=args.moco_t,
                      num_splits=args.num_splits,
                      lr=args.lr,
                      momentum=args.momentum,
                      weight_decay=args.weight_decay)
    print(model)

    #### augmentation as a collate_fn
    collate_fn = lightly.data.collate.SimCLRCollateFunction(input_size=96, gaussian_blur=0.1)

    #### train dataloader
    base = CustomDataset(args.data, "unlabeled", None) # use empty transform
    train_dataset = lightly.data.LightlyDataset.from_torch_dataset(base)
    train_loader = torch.utils.data.DataLoader(train_dataset, 
                                                batch_size=args.batch_size, 
                                                shuffle=True,
                                                num_workers=args.workers, 
                                                pin_memory=True, 
                                                drop_last=True,
                                                collate_fn=collate_fn)

    #### set training set ratio
    if args.small_set:
        print('=> Using 1/10 unlabeled set')
        overfit_batches = 0.1
    else:
        print('=> Using full unlabeled set')
        overfit_batches = 0.0

    #### resume checkpoints
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint_path = args.resume
    else:
        checkpoint_path = None

    #### save checkpoint path
    checkpoint_callback = ModelCheckpoint(dirpath=args.checkpoint_dir,
                                          filename='pretrain_{epoch:03d}_{loss:.2f}',
                                          period=args.save_checkpoint_per_epoch)

    trainer = pl.Trainer(max_epochs=args.epochs, 
                         gpus=args.gpus,
                         progress_bar_refresh_rate=100,
                         overfit_batches=overfit_batches,
                         resume_from_checkpoint=checkpoint_path,
                         benchmark=True,
                        #  auto_lr_find=True,
                         callbacks=[checkpoint_callback]) 

    print("=> Start training")
    trainer.fit(model=model, train_dataloader=train_loader)


if __name__ == '__main__':
    main()
