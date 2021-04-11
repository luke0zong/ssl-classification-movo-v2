# Author: Arthur Jinyue Guo (jg5505).
import torch
from pytorch import nn
import pytorch.nn.functional as F

class BasicBlock(nn.Module):
    """
    A basic block of resnet.

    Parameters
    ----------
    layer_name: str
        e.g. "conv_2
    in_channel: int
        input channel of the entire block.
    out_channels: list of ints
        output channels of each layer. e.g [64, 64, 256] as conv2_x of resnet
    kernel_sizes: list of ints
        kernel sizes of each layer. e.g [1, 3, 1] as conv2_x of resnet
    number: int
        number of blocks. e.g. 3 as conv2_x.
    """

    def __init__(self, in_channel, out_channels, kernel_sizes, stride=1):
        super(BasicBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channel, out_channels[0],
                                kernel_size=kernel_sizes[0],
                                padding=(kernel_sizes[0]-1)/2,
                                stride=stride,
                                bias=False))

        self.bn1 = nn.BatchNorm2d(out_channels[0]))

        self.conv2 = nn.Conv2d(out_channels[0], out_channels[1],
                                kernel_size=kernel_sizes[1],
                                padding=(kernel_sizes[1]-1)/2,
                                stride=1,
                                bias=False))

        self.bn2 = nn.BatchNorm2d(out_channels[1]))

        self.conv3 = nn.Conv2d(out_channels[1], out_channels[2],
                                kernel_size=kernel_sizes[2],
                                padding=(kernel_sizes[2]-1)/2,
                                stride=1,
                                bias=False))

        self.bn3 = nn.BatchNorm2d(out_channels[2]))

        # if stride==1, use identity residual
        self.shortcut = nn.Sequential()
        # otherwise, use a 1x1 conv layer to match the output shape
        if stride != 1:
            self.shortcut = nn.Sequential(
                                        nn.Conv2d(
                                            in_channel,
                                            out_channels[2],
                                            kernel_size=1,
                                            stride=stride,
                                            padding=0,
                                            bias=False), 
                                        nn.BatchNorm2d(out_channels))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Resnet50(nn.Module):
    """
    A resnet50 model with input shape of 3 x 96 x 96.
    """

    def __init__(self, block, num_blocks, num_classes=800):
        super(Resnet50, self).__init__()

        # in shape: (N, 3, 96, 96)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        # out shape: (N, 64, 48, 48)
        self.conv2_1 = BasicBlock(64, [64,64,256], [1,3,1], 2)
        self.conv2_2 = BasicBlock(256, [64,64,256], [1,3,1], 1)
        self.conv2_3 = BasicBlock(256, [64,64,256], [1,3,1], 1)
        # out shape: (N, 256, 24, 24)
        self.conv3_1 = BasicBlock(256, [128,128,512], [1,3,1], 2)
        self.conv3_2 = BasicBlock(512, [128,128,512], [1,3,1], 1)
        self.conv3_3 = BasicBlock(512, [128,128,512], [1,3,1], 1)
        self.conv3_4 = BasicBlock(512, [128,128,512], [1,3,1], 1)
        # out shape: (N, 512, 12, 12)
        self.conv4_1 = BasicBlock(512, [256, 256, 1024], [1,3,1], 2)
        self.conv4_2 = BasicBlock(1024), [256, 256, 1024], [1,3,1], 1)
        self.conv4_3 = BasicBlock(1024), [256, 256, 1024], [1,3,1], 1)
        self.conv4_4 = BasicBlock(1024), [256, 256, 1024], [1,3,1], 1)
        self.conv4_5 = BasicBlock(1024), [256, 256, 1024], [1,3,1], 1)
        self.conv4_6 = BasicBlock(1024), [256, 256, 1024], [1,3,1], 1)
        # out shape: (N, 2014, 6, 6)
        self.conv5_1 = BasicBlock(1024, [512, 512, 2048], [1,3,1], 2)
        self.conv5_2 = BasicBlock(2048, [512, 512, 2048], [1,3,1], 2)
        self.conv5_3 = BasicBlock(2048, [512, 512, 2048], [1,3,1], 2)
        # out shape: (N, 2048, 3, 3)
        # flatten to (N, 2048*3*3)
        self.fc = nn.Linear(2048*3*3, num_classes)
        # logsoftargmax

    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))

        out = self.conv2_1(out)
        out = self.conv2_2(out)
        out = self.conv2_3(out)

        out = self.conv3_1(out)
        out = self.conv3_2(out)
        out = self.conv3_3(out)
        out = self.conv3_4(out)

        out = self.conv4_1(out)
        out = self.conv4_2(out)
        out = self.conv4_3(out)
        out = self.conv4_4(out)
        out = self.conv4_5(out)
        out = self.conv4_6(out)

        out = self.conv5_1(out)
        out = self.conv5_2(out)
        out = self.conv5_3(out)
        
        out = torch.flatten(out, 1)
        out = self.fc(out)

        return F.log_softmax(out, dim=1)
