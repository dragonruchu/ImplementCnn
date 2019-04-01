'''DenseNet in Pytorch'''

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class Bottleneck(nn.Module):
    def __init__(self, in_channels, growth_gate):
        super(Bottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, 4 * growth_gate, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(4 * growth_gate)
        self.conv2 = nn.Conv2d(4 * growth_gate, growth_gate, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        out = torch.cat([out, x], 1)
        return out


class Transition(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Transition, self).__init__()
        self.bn = nn.BatchNorm2d(in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, x):
        out = self.conv(F.relu(self.bn(x)))
        out = F.adaptive_avg_pool2d(out, 2)
        return out


class DenseNet(nn.Module):
    """docstring for DenseNet"""

    def __init__(self, block, nblocks, growth_gate=12, reduction=0.5, num_classes=10):
        super(DenseNet, self).__init__()
        self.growth_gate = growth_gate

        num_channels = 2 * growth_gate
        self.conv1 = nn.Conv2d(3, num_channels, kernel_size=1, padding=1, bias=False)

        self.dense1 = self._make_dense_layers(block, num_channels, nblocks[0])
        num_channels += nblocks[0] * growth_gate
        out_channels = int(math.floor(num_channels * reduction))
        self.trans1 = Transition(num_channels, out_channels)
        num_channels = out_channels

        self.dense2 = self._make_dense_layers(block, num_channels, nblocks[1])
        num_channels += nblocks[1] * growth_gate
        out_channels = int(math.floor(num_channels * reduction))
        self.trans2 = Transition(num_channels, out_channels)
        num_channels = out_channels

        self.dense3 = self._make_dense_layers(block, num_channels, nblocks[2])
        num_channels += nblocks[2] * growth_gate
        out_channels = int(math.floor(num_channels * reduction))
        self.trans3 = Transition(num_channels, out_channels)
        num_channels = out_channels

        self.dense4 = self._make_dense_layers(block, num_channels, nblocks[3])
        num_channels += nblocks[3] * growth_gate

        self.bn = nn.BatchNorm2d(num_channels)
        self.linear = nn.Linear(num_channels, num_classes)

    def _make_dense_layers(self, block, in_channels, nblock):
        layers = []
        for i in range(nblock):
            layers.append(block(in_channels, self.growth_gate))
            in_channels += self.growth_gate
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.trans1(self.dense1(out))
        out = self.trans2(self.dense2(out))
        out = self.trans3(self.dense3(out))
        out = self.dense4(out)
        out = F.avg_pool2d(F.relu(self.bn(out)), 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)

        return out


def DenseNet121():
    return DenseNet(Bottleneck, [6, 12, 24, 16], growth_gate=32)


def test():
    net = DenseNet121()
    print(net)
    x = torch.randn(1, 3, 32, 32)
    y = net(x)
    print(y)


# test()
