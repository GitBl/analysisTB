"""
Network and module definition
"""

import numpy as np
import torch
import torch.nn as nn
import torchvision.datasets
from torchvision import transforms
import matplotlib.pyplot as plt
import torch.nn.functional as F


class ResConv(nn.Module):
    """
    Define a Residual Convolutional unit
    """

    def __init__(self, channel, residual=True, batch_norm=False):
        super(ResConv, self).__init__()
        self.residual = residual
        self.conv = nn.Conv2d(channel, channel, 3, padding=1)
        self.batch_norm_enabled = batch_norm

        if(self.batch_norm_enabled):
            self.bn = nn.BatchNorm2d(channel)

    def forward(self, x):
        output = F.relu(self.conv(x))
        if(self.batch_norm_enabled):
            output = self.bn(output)
        if(self.residual):
            output = output + x
        return output


class GenResNet(nn.Module):
    """
    Generate a range of convolutional units
    """

    def __init__(self, depth, width, residual=True, batch_norm=False):
        self.width = width
        super(GenResNet, self).__init__()
        self.first_conv = nn.Conv2d(3, width, 3, padding=1)
        self.first_conv.first = True
        layers = []
        for i in range(depth):
            layers += [ResConv(width, residual, batch_norm)]
        self.multi_conv = nn.Sequential(*layers)
        self.lin = nn.Linear(width*32*32, 10)

    def forward(self, x):
        x = self.first_conv(x)
        x = self.multi_conv(x)
        x = x.view(-1, self.width*32*32)
        x = self.lin(x)
        return x


class SimpleResNet(nn.Module):  # Taken from HW3
    def __init__(self):
        super(SimpleResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.pool1 = nn.MaxPool2d(2, stride=2)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv4 = nn.Conv2d(64, 64, 3, padding=1)  # tagged as end-like
        self.conv5 = nn.Conv2d(64, 64, 3, padding=1)  # tagged as end-like
        self.pool2 = nn.MaxPool2d(2, stride=2)
        self.conv6 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv7 = nn.Conv2d(128, 128, 3, padding=1)  # tagged as end-like
        self.conv8 = nn.Conv2d(128, 128, 3, padding=1)  # tagged as end-like
        self.avgpool = nn.AvgPool2d(kernel_size=8, stride=8)
        self.lin = nn.Linear(128*1*1, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool1(x)
        x = F.relu(self.conv3(x))
        x_state_1 = x
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x) + x_state_1)
        x = self.pool2(x)
        x = F.relu(self.conv6(x))
        x_state_2 = x
        x = F.relu(self.conv7(x))
        x = F.relu(self.conv8(x) + x_state_2)
        x = self.avgpool(x)
        x = x.view(-1, 128)
        x = self.lin(x)
        return x
