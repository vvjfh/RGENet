"""
3D U-Net_Learning Dense Volumetric Segmentation from Sparse Annotation
3D U-Net:从稀疏标注学习密集体积分割
"""

import torch
import torch.nn as nn
import sys
from math import sqrt


class DoubleConv(nn.Module):
    def __init__(self, in_channel, out_channel):
        """
        双层3x3卷积单元（感受野：5）
        :param in_channel: 输入通道
        :param out_channel: 输出通道
        """
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channel, out_channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(out_channel),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv3d(out_channel, out_channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(out_channel),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


# G：3D U-Net
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2, padding=0)

        # Encoder
        self.E1 = nn.Sequential(
            DoubleConv(1, 32),
        )
        self.E2 = nn.Sequential(
            DoubleConv(32, 64),
        )
        self.E3 = nn.Sequential(
            DoubleConv(64, 128),
        )
        self.E4 = nn.Sequential(
            DoubleConv(128, 256),
        )

        self.bottleneck = DoubleConv(256, 512)

        # Decoder
        self.up1 = nn.Sequential(
            nn.ConvTranspose3d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
        )
        self.D1 = DoubleConv(512, 256)

        self.up2 = nn.Sequential(
            nn.ConvTranspose3d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
        )
        self.D2 = DoubleConv(256, 128)

        self.up3 = nn.Sequential(
            nn.ConvTranspose3d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
        )
        self.D3 = DoubleConv(128, 64)

        self.up4 = nn.Sequential(
            nn.ConvTranspose3d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
        )
        self.D4 = DoubleConv(64, 32)

        self.final_conv = nn.Sequential(
            nn.Conv3d(32, 1, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x):
        """
        :param x: 三维数据输入，注意尺寸在各级卷积之后需满足2的倍数
        :return: 预测分割图
        """
        # -------------------------------- Encoder --------------------------------
        e1 = self.E1(x)
        pool1 = self.pool(e1)

        e2 = self.E2(pool1)
        pool2 = self.pool(e2)

        e3 = self.E3(pool2)
        pool3 = self.pool(e3)

        e4 = self.E4(pool3)
        pool4 = self.pool(e4)

        e5 = self.bottleneck(pool4)

        # -------------------------------- Decoder --------------------------------
        d1 = self.up1(e5)
        d1 = torch.cat((d1, e4), dim=1)
        d1 = self.D1(d1)

        d2 = self.up2(d1)
        d2 = torch.cat((d2, e3), dim=1)
        d2 = self.D2(d2)

        d3 = self.up3(d2)
        d3 = torch.cat((d3, e2), dim=1)
        d3 = self.D3(d3)

        d4 = self.up4(d3)
        d4 = torch.cat((d4, e1), dim=1)
        d4 = self.D4(d4)

        # 输出分割图
        out = self.final_conv(d4)

        return out


if __name__ == '__main__':
    images = torch.randn(2, 1, 112, 112, 112)
    gt = torch.randn(2, 1, 112, 112, 112)

    generator = Generator()
    preds = generator(images)

    print(preds.shape)
