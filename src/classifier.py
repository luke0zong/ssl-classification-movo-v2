import torch
import torch.nn as nn
import torchvision
import pytorch_lightning as pl
import lightly

class Classifier(pl.LightningModule):
    def __init__(self, model, lr):
        super().__init__()
        # create a moco based on ResNet
        # self.resnet_moco = model
        self.resnet = model.resnet_moco.backbone
        self.lr = lr

        # freeze the layers of moco
        for p in self.resnet.parameters():  # reset requires_grad
            p.requires_grad = False

        self.fc = nn.Linear(512, 800)

        self.accuracy = pl.metrics.Accuracy()

    def forward(self, x):
        with torch.no_grad():
            y_hat = self.resnet(x).squeeze()
            # y_hat = nn.functional.normalize(y_hat, dim=1)
            y_hat = nn.functional.argmax(y_hat, dim=1)
        y_hat = self.fc(y_hat)
        return y_hat

    # We provide a helper method to log weights in tensorboard
    # which is useful for debugging.
    # def custom_histogram_weights(self):
    #     for name, params in self.named_parameters():
    #         self.logger.experiment.add_histogram(
    #             name, params, self.current_epoch)

    def training_step(self, batch, batch_idx):
        x, y, _ = batch
        y_hat = self.forward(x)
        loss = nn.functional.cross_entropy(y_hat, y)
        self.log('train_loss_fc', loss)
        return loss

    # def training_epoch_end(self, outputs):
    #     self.custom_histogram_weights()

    def validation_step(self, batch, batch_idx):
        x, y, _ = batch
        y_hat = self.forward(x)
        self.accuracy(y_hat, y)
        self.log('val_acc', self.accuracy.compute(),
                 on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        optim = torch.optim.SGD(self.fc.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, 100)
        return [optim], [scheduler]
