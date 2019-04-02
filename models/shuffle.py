'''ShuffleNet in PyTorch.
Reference:
[1] ShuffleNet: An Extremely Efficient Convolutional Neural Network for Mobile Devices
[2] https://github.com/kuangliu/pytorch-cifar
'''

import torch
import torch.nn as nn
import torch.nn.functional as F


class ShuffleBlock(nn.Module):
    def __init__(self, groups):
        super(ShuffleBlock, self).__init__()
        self.groups = groups

    def forward(self, x):
        N, C, H, W = x.size()
        g = self.groups
        # 将channels先分后重排再合起来
        # print(x.view(N, g, int(C / g), H, W).permute(0, 2, 1, 3, 4).contiguous().view(N, C, H, W).shape)
        return x.view(N, g, int(C / g), H, W).permute(0, 2, 1, 3, 4).contiguous().view(N, C, H, W)


class Bottleneck(nn.Module):
    """docstring for Bottleneck"""

    def __init__(self, in_channels, out_channels, stride, groups):
        super(Bottleneck, self).__init__()
        self.stride = stride

        mid_channels = int(out_channels / 4)
        g = 1 if in_channels == 24 else groups
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1, groups=g,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.shuffle1 = ShuffleBlock(groups=g)
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=stride,
                               padding=1, groups=mid_channels, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_channels)
        self.conv3 = nn.Conv2d(mid_channels, out_channels, kernel_size=1, groups=groups, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride == 2:
            self.shortcut = nn.Sequential(nn.AvgPool2d(3, stride=2, padding=1))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.shuffle1(out)
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        res = self.shortcut(x)

        out = F.relu(torch.cat([out, res], 1)) if self.stride == 2 else F.relu(out + res)

        return out


class ShuffleNet(nn.Module):
    """docstring for ShuffleNet"""

    def __init__(self, cfg):
        super(ShuffleNet, self).__init__()
        out_channels = cfg['out_channels']
        num_blocks = cfg['num_blocks']
        groups = cfg['groups']

        self.conv1 = nn.Conv2d(3, 24, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(24)
        self.in_channels = 24
        self.layer1 = self._make_layer(out_channels[0], num_blocks[0], groups)
        self.layer2 = self._make_layer(out_channels[1], num_blocks[1], groups)
        self.layer3 = self._make_layer(out_channels[2], num_blocks[2], groups)
        self.linear = nn.Linear(out_channels[2], 10)

    def _make_layer(self, out_channels, num_blocks, groups):
        layers = []
        for i in range(num_blocks):
            stride = 2 if i == 0 else 1
            cat_channels = self.in_channels if i == 0 else 0
            layers.append(Bottleneck(self.in_channels, out_channels - cat_channels, stride=stride, groups=groups))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ShuffleNeG2():
    cfg = {
        'out_channels': [200, 400, 800],
        'num_blocks': [4, 8, 4],
        'groups': 2
    }
    return ShuffleNet(cfg)


def test():
    net = ShuffleNeG2()
    print(net)
    x = torch.randn(1, 3, 32, 32)
    y = net(x)
    print(y)


test()
