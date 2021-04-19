# Author: Arthur Jinyue Guo (jg5505)
import torch
import torch.nn.functional as F
from torch import Tensor
from torch import nn
from typing import Type, Any, Callable, Union, List, Optional


class SubBatchNorm2d(nn.Module):
    """
    The standard BN layer computes stats across all examples in a GPU. In some
    cases it is desirable to compute stats across only a subset of examples
    (e.g., in multigrid training https://arxiv.org/abs/1912.00998).
    SubBatchNorm2d splits the batch dimension into N splits, and run BN on
    each of them separately (so that the stats are computed on each subset of
    examples (1/N of batch) independently. During evaluation, it aggregates
    the stats from all splits into one BN.
    """

    def __init__(self, num_features, num_splits=2, **args):
        """
        Args:
            num_splits (int): number of splits.
            args (list): other arguments.
        """
        super(SubBatchNorm2d, self).__init__()
        self.num_splits = num_splits
        self.bn = nn.BatchNorm2d(num_features, **args)
        self.split_bn = nn.BatchNorm2d(num_features * num_splits, **args)

    def _get_aggregated_mean_std(self, means, stds, n):
        """
        Calculate the aggregated mean and stds.
        Args:
            means (tensor): mean values.
            stds (tensor): standard deviations.
            n (int): number of sets of means and stds.
        """
        mean = means.view(n, -1).sum(0) / n
        std = (
            stds.view(n, -1).sum(0) / n
            + ((means.view(n, -1) - mean) ** 2).view(n, -1).sum(0) / n
        )
        return mean.detach(), std.detach()

    def aggregate_stats(self):
        """
        Synchronize running_mean, and running_var. Call this before eval.
        """
        if self.split_bn.track_running_stats:
            (
                self.bn.running_mean.data,
                self.bn.running_var.data,
            ) = self._get_aggregated_mean_std(
                self.split_bn.running_mean,
                self.split_bn.running_var,
                self.num_splits,
            )

    def forward(self, x):
        if self.training:
            n, c, h, w = x.shape
            x = x.view(n // self.num_splits, c * self.num_splits, h, w)
            x = self.split_bn(x)
            x = x.view(n, c, h, w)
        else:
            x = self.bn(x)
        return x


def test():
    """
    Simple test for resnet50 class.
    """
    print("Buiding network")
    net = Resnet50()
    net.cuda()
    x = torch.randn([64, 3, 96, 96]).cuda()
    print("Running random input")
    y = net(x)
    print("resnet50 simple test succeed!")


if __name__ == '__main__':
    test()