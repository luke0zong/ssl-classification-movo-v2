# Feel free to modify this file.

import torch
from torchvision import models, transforms
from torch import nn

team_id = 9
team_name = "Bai Ze"
email_address = "jg5505@nyu.edu"


class SubBatchNorm2d(nn.BatchNorm2d):
    """Simulates multi-gpu behaviour of BatchNorm in one gpu by splitting.
    Implementation was adapted from:
    https://github.com/davidcpage/cifar10-fast/blob/master/torch_backend.py
    Attributes:
        num_features:
            Number of input features.
        num_splits:
            Number of splits.
    """

    def __init__(self, num_features, num_splits=2, **kw):
        super().__init__(num_features, **kw)
        self.num_splits = num_splits
        self.register_buffer(
            'running_mean', torch.zeros(num_features*self.num_splits)
        )
        self.register_buffer(
            'running_var', torch.ones(num_features*self.num_splits)
        )

    def train(self, mode=True):
        # lazily collate stats when we are going to use them
        if (self.training is True) and (mode is False):
            self.running_mean = \
                torch.mean(
                    self.running_mean.view(self.num_splits, self.num_features),
                    dim=0
                ).repeat(self.num_splits)
            self.running_var = \
                torch.mean(
                    self.running_var.view(self.num_splits, self.num_features),
                    dim=0
                ).repeat(self.num_splits)

        return super().train(mode)

    def forward(self, input):
        """Computes the SplitBatchNorm on the input.
        """
        # get input shape
        N, C, H, W = input.shape

        # during training, use different stats for each split and otherwise
        # use the stats from the first split
        if self.training or not self.track_running_stats:
            result = nn.functional.batch_norm(
                input.view(-1, C*self.num_splits, H, W),
                self.running_mean, self.running_var,
                self.weight.repeat(self.num_splits),
                self.bias.repeat(self.num_splits),
                True,
                self.momentum,
                self.eps
            ).view(N, C, H, W)
        else:
            result = nn.functional.batch_norm(
                input,
                self.running_mean[:self.num_features],
                self.running_var[:self.num_features],
                self.weight,
                self.bias,
                False,
                self.momentum,
                self.eps
            )

        return result


def get_model():
    model = models.resnet50(num_classes=800, norm_layer=SubBatchNorm2d)
    print(model)
    return model


# Transforms
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
eval_transform = transforms.Compose([
    transforms.Resize(128),  # add resize
    transforms.CenterCrop(96),  # add crop
    transforms.ToTensor(),
    normalize
])
