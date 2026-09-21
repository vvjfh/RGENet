import torch
import torch.nn as nn
import torch.nn.functional as F
import torchsummary
from torchsummary.torchsummary import summary


class Basic_blocks(nn.Module):
    def __init__(self, in_channel, out_channel, decay=1):
        super(Basic_blocks, self).__init__()
        self.conv = nn.Conv3d(in_channel, out_channel, 1)
        self.conv1 = nn.Sequential(
            nn.Conv3d(out_channel, out_channel, 3, padding=1),
            nn.InstanceNorm3d(out_channel, affine=True, track_running_stats=False),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv3d(out_channel, out_channel, 3, padding=1),
            nn.InstanceNorm3d(out_channel, affine=True, track_running_stats=False),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        return conv2 + x


class DSE(nn.Module):
    def __init__(self, in_channel, decay=2):
        super(DSE, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv3d(in_channel, in_channel // decay, 1),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channel // decay, in_channel, 1),
            nn.Sigmoid()
        )
        self.layer2 = nn.Sequential(
            nn.Conv3d(in_channel, in_channel // decay, 1),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channel // decay, in_channel, 1),
            nn.Sigmoid()
        )
        self.gpool = nn.AdaptiveAvgPool3d(1)
        self.gapool = nn.AdaptiveMaxPool3d(1)

    def forward(self, x):
        gp = self.gpool(x)
        se = self.layer1(gp)
        x = x * se
        gap = self.gapool(x)
        se2 = self.layer2(gap)
        return x * se2


class Spaceatt(nn.Module):
    def __init__(self, in_channel, decay=2):
        super(Spaceatt, self).__init__()
        self.Q = nn.Sequential(
            nn.Conv3d(in_channel, in_channel // decay, 1),
            nn.InstanceNorm3d(in_channel // decay, affine=True, track_running_stats=False),
            nn.Conv3d(in_channel // decay, 1, 1),
            nn.Sigmoid()
        )
        self.K = nn.Sequential(
            nn.Conv3d(in_channel, in_channel // decay, 3, padding=1),
            nn.InstanceNorm3d(in_channel // decay, affine=True, track_running_stats=False),
            nn.Conv3d(in_channel // decay, in_channel // decay, 3, padding=1),
            DSE(in_channel // decay)
        )
        self.V = nn.Sequential(
            nn.Conv3d(in_channel, in_channel // decay, 3, padding=1),
            nn.InstanceNorm3d(in_channel // decay, affine=True, track_running_stats=False),
            nn.Conv3d(in_channel // decay, in_channel // decay, 3, padding=1),
            DSE(in_channel // decay)
        )
        self.sig = nn.Sequential(
            nn.Conv3d(in_channel // decay, in_channel, 3, padding=1),
            nn.InstanceNorm3d(in_channel, affine=True, track_running_stats=False),
            nn.ReLU(inplace=True)
        )

    def forward(self, low, high):
        Q = self.Q(low)
        K = self.K(low)
        V = self.V(high)
        att = Q * K
        # 3D数据不能直接使用矩阵乘法，这里用元素乘法替代
        att = att * V
        return self.sig(att)


class CSCA_blocks(nn.Module):
    def __init__(self, in_channel, out_channel, decay=2):
        super(CSCA_blocks, self).__init__()
        self.upsample = nn.ConvTranspose3d(
            in_channel, out_channel, 2, stride=2)
        self.conv = Basic_blocks(in_channel, out_channel // 2)
        self.catt = DSE(out_channel // 2, decay)
        self.satt = Spaceatt(out_channel // 2, decay)

    def forward(self, high, low):
        up = self.upsample(high)
        # 处理奇数尺寸导致的上采样与跳连特征空间尺寸不一致问题
        if up.shape[2:] != low.shape[2:]:
            up = F.interpolate(up, size=low.shape[2:], mode='trilinear', align_corners=False)
        concat = torch.cat([up, low], dim=1)
        point = self.conv(concat)
        catt = self.catt(point)
        satt = self.satt(point, catt)
        plusatt = catt * satt
        return torch.cat([plusatt, catt], dim=1)


class CSCAUNet3D(nn.Module):
    def __init__(self, n_class=1, decay=2, in_channels=1):
        super(CSCAUNet3D, self).__init__()
        self.pool = nn.MaxPool3d(2)

        self.down_conv1 = Basic_blocks(in_channels, 32, decay)
        self.down_conv2 = Basic_blocks(32, 64, decay)
        self.down_conv3 = Basic_blocks(64, 128, decay)
        self.down_conv4 = Basic_blocks(128, 256, decay)
        self.down_conv5 = Basic_blocks(256, 512, decay)

        self.down_conv6 = nn.Sequential(
            Basic_blocks(512, 1024, decay),
            DSE(1024, decay)
        )

        self.up_conv5 = CSCA_blocks(1024, 512, decay)
        self.up_conv4 = CSCA_blocks(512, 256, decay)
        self.up_conv3 = CSCA_blocks(256, 128, decay)
        self.up_conv2 = CSCA_blocks(128, 64, decay)
        self.up_conv1 = CSCA_blocks(64, 32, decay)

        self.dp6 = nn.Conv3d(1024, 1, 1)
        self.dp5 = nn.Conv3d(512, 1, 1)
        self.dp4 = nn.Conv3d(256, 1, 1)
        self.dp3 = nn.Conv3d(128, 1, 1)
        self.dp2 = nn.Conv3d(64, 1, 1)
        self.out = nn.Conv3d(32, 1, 3, padding=1)

        self.center5 = nn.Conv3d(1024, 512, 1)
        self.decodeup4 = nn.Conv3d(512, 256, 1)
        self.decodeup3 = nn.Conv3d(256, 128, 1)
        self.decodeup2 = nn.Conv3d(128, 64, 1)

    def forward(self, inputs):
        b, c, d, h, w = inputs.size()

        down1 = self.down_conv1(inputs)
        pool1 = self.pool(down1)

        down2 = self.down_conv2(pool1)
        pool2 = self.pool(down2)

        down3 = self.down_conv3(pool2)
        pool3 = self.pool(down3)

        down4 = self.down_conv4(pool3)
        pool4 = self.pool(down4)

        down5 = self.down_conv5(pool4)
        pool5 = self.pool(down5)

        center = self.down_conv6(pool5)

        out6 = self.dp6(center)
        out6 = F.interpolate(
            out6, (d, h, w), mode='trilinear', align_corners=False)

        deco5 = self.up_conv5(center, down5)
        out5 = self.dp5(deco5)
        out5 = F.interpolate(
            out5, (d, h, w), mode='trilinear', align_corners=False)
        center5 = self.center5(center)
        center5 = F.interpolate(center5, (d // 16, h // 16, w // 16),
                                mode='trilinear', align_corners=False)
        deco5 = deco5 + center5

        deco4 = self.up_conv4(deco5, down4)
        out4 = self.dp4(deco4)
        out4 = F.interpolate(
            out4, (d, h, w), mode='trilinear', align_corners=False)
        decoderup4 = self.decodeup4(deco5)
        decoderup4 = F.interpolate(
            decoderup4, (d // 8, h // 8, w // 8), mode='trilinear', align_corners=False)
        deco4 = deco4 + decoderup4

        deco3 = self.up_conv3(deco4, down3)
        out3 = self.dp3(deco3)
        out3 = F.interpolate(
            out3, (d, h, w), mode='trilinear', align_corners=False)
        decoderup3 = self.decodeup3(deco4)
        decoderup3 = F.interpolate(
            decoderup3, (d // 4, h // 4, w // 4), mode='trilinear', align_corners=False)
        deco3 = deco3 + decoderup3

        deco2 = self.up_conv2(deco3, down2)
        out2 = self.dp2(deco2)
        out2 = F.interpolate(out2, (d, h, w), mode='trilinear', align_corners=False)
        decoderup2 = self.decodeup2(deco3)
        decoderup2 = F.interpolate(
            decoderup2, (d // 2, h // 2, w // 2), mode='trilinear', align_corners=False)
        deco2 = deco2 + decoderup2

        deco1 = self.up_conv1(deco2, down1)
        out = self.out(deco1)

        return out, out2, out3, out4, out5, out6


if __name__ == '__main__':
    # 注意这里修改了输入通道为1（PET数据通常是单通道），调整尺寸为112x112x112
    model = CSCAUNet3D(1, 2, in_channels=1)
    summary(model, (1, 112, 112, 112), batch_size=1, device='cpu')
