import torch
import torch.nn as nn
import torchvision
import pytorch_lightning as pl
import lightly

class MocoModel(pl.LightningModule):
    def __init__(self, moco_dim, moco_k, moco_m, moco_t, num_splits, lr, momentum, weight_decay):
        super().__init__()
        self.moco_dim = moco_dim
        self.moco_k = moco_k
        self.moco_m = moco_m
        self.moco_t = moco_t
        self.num_splits = num_splits
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay

        # create a ResNet backbone and remove the classification head
        resnet = lightly.models.ResNetGenerator('resnet-18', num_splits=self.num_splits)
        backbone = nn.Sequential(
            *list(resnet.children())[:-1],
            nn.AdaptiveAvgPool2d(1),
        )

        # create a moco based on ResNet
        self.resnet_moco = lightly.models.MoCo(backbone,
                                                num_ftrs=512, # fixed for resnet-18
                                                out_dim=self.moco_dim,
                                                m=self.moco_m,
                                                batch_shuffle=True)

        # create our loss with the optional memory bank
        self.criterion = lightly.loss.NTXentLoss(temperature=self.moco_t, memory_bank_size=self.moco_k)

    def forward(self, x):
        self.resnet_moco(x)

    # We provide a helper method to log weights in tensorboard
    # which is useful for debugging.
    # def custom_histogram_weights(self):
    #     for name, params in self.named_parameters():
    #         self.logger.experiment.add_histogram(
    #             name, params, self.current_epoch)

    def training_step(self, batch, batch_idx):
        (x0, x1), _, _ = batch
        y0, y1 = self.resnet_moco(x0, x1)
        loss = self.criterion(y0, y1)
        self.log('train_loss_ssl', loss)
        return loss

    # def training_epoch_end(self, outputs):
    #     self.custom_histogram_weights()

    def configure_optimizers(self):
        optim = torch.optim.SGD(self.resnet_moco.parameters(), 
                                lr=self.lr,
                                momentum=self.momentum, 
                                weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, max_epochs)
        return [optim], [scheduler]