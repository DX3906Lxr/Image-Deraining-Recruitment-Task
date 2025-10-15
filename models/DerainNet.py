import torch
import torch.nn as nn
import torch.nn.functional as F

class Conv3x3(nn.Module):
    def __init__(self, in_ch, out_ch, bias=True):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        return self.conv(x)


class ResidualBlock(nn.Module):

    def __init__(self, ch):
        super().__init__()
        self.conv1 = Conv3x3(ch, ch)
        self.act   = nn.ReLU(inplace=True)
        self.conv2 = Conv3x3(ch, ch)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.act(out)
        out = self.conv2(out)
        return out + identity

class ResidualStack(nn.Module):
    def __init__(self, ch, n_blocks=8):
        super().__init__()
        self.blocks = nn.Sequential(*[ResidualBlock(ch) for _ in range(n_blocks)])

    def forward(self, x):
        return self.blocks(x)


class UpsampleBlock(nn.Module):
    def __init__(self, channels, scale_factor=2):
        super().__init__()
        self.scale_factor = scale_factor
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.act   = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode="bilinear", align_corners=False)
        x = self.conv1(x)
        x = self.act(x)
        x = self.conv2(x)
        return x

class DerainBranch(nn.Module):

    def __init__(self, ch, scale=1, n_blocks=8):

        super().__init__()

        self.scale = scale
        if scale == 1:
            self.down = nn.Identity()
        elif scale == 2:
            self.down = nn.MaxPool2d(kernel_size=2, stride=2)
        else:
            # 连续两次 2x 下采样，避免一次性 4x 带来的信息损失过猛
            self.down = nn.Sequential(
                nn.MaxPool2d(2, 2),
                nn.MaxPool2d(2, 2),
            )

        self.body = ResidualStack(ch, n_blocks=n_blocks)

        if scale == 1:
            self.upsample = nn.Identity()
        elif scale == 2:
            self.upsample = UpsampleBlock(ch, scale_factor=2)
        else:  # scale == 4
            # 分两次上采样，每次 2 倍
            self.upsample = nn.Sequential(
                UpsampleBlock(ch, scale_factor=2),
                UpsampleBlock(ch, scale_factor=2),
            )

    def forward(self, x, out_size_hw):
        x_down = self.down(x)
        feat   = self.body(x_down) + x_down
        out = self.upsample(feat)
        return out

class DerainNet(nn.Module):

    def __init__(self, base_channels=64, n_blocks_per_branch=8):
        super().__init__()
        C = base_channels

        self.stem = nn.Sequential(
            Conv3x3(3, C),
            nn.ReLU(inplace=True)
        )

        self.branch1 = DerainBranch(C, scale=1, n_blocks=n_blocks_per_branch)  # 原图尺度
        self.branch2 = DerainBranch(C, scale=2, n_blocks=n_blocks_per_branch)  # 1/2
        self.branch3 = DerainBranch(C, scale=4, n_blocks=n_blocks_per_branch)  # 1/4

        self.fuse = nn.Sequential(
            Conv3x3(3 * C, C),
            nn.ReLU(inplace=True),
            Conv3x3(C, 3)
        )

    def forward(self, x):
        B, _, H, W = x.shape
        feat = self.stem(x)

        b1 = self.branch1(feat, (H, W))
        b2 = self.branch2(feat, (H, W))
        b3 = self.branch3(feat, (H, W))

        fused = torch.cat([b1, b2, b3], dim=1)
        rain  = self.fuse(fused)

        derained = x - rain
        return derained

