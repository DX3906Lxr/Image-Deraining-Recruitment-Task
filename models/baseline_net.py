import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, channels=64):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = out + x
        out = self.relu(out)
        return out

class BaselineNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, num_features=64):
        super(BaselineNet, self).__init__()
        self.head = nn.Conv2d(in_channels, num_features, kernel_size=3, stride=1, padding=1)
        self.resblocks = nn.Sequential(
            *[ResidualBlock(num_features) for _ in range(4)]
        )
        self.tail = nn.Conv2d(num_features, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        out = self.head(x)
        out = self.resblocks(out)
        out = self.tail(out)
        clean = x - out

        return clean


