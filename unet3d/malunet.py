"""
3D U-Net_Learning Dense Volumetric Segmentation from Sparse Annotation
3D U-Net:从稀疏标注学习密集体积分割
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(Conv3d => BN => ReLU) * 2"""
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class IEAblock(nn.Module):
    def __init__(self, in_c):
        super(IEAblock, self).__init__()
        self.conv1 = nn.Conv3d(in_c, in_c, 1)
        self.k = in_c * 4
        self.conv2 = nn.Conv3d(in_c, in_c, 1, bias=False)
        self.norm_layer = nn.GroupNorm(4, in_c)

    def forward(self, x):
        idn = x
        x = self.conv1(x)
        b, c, d, h, w = x.size()
        x = x.view(b, c, d * h * w)  # [B, C, N]
        # 动态创建Conv1d，输入通道c，输出通道k
        linear_0 = nn.Conv1d(c, self.k, 1, bias=False).to(x.device)
        linear_1 = nn.Conv1d(self.k, c, 1, bias=False).to(x.device)
        attn = linear_0(x)
        attn = F.softmax(attn, dim=-1)
        attn = attn / (1e-9 + attn.sum(dim=1, keepdim=True))
        x = linear_1(attn)
        x = x.view(b, c, d, h, w)
        x = self.norm_layer(self.conv2(x))
        x = x + idn
        x = F.gelu(x)
        return x


class DepthwiseSeparableConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1):
        super(DepthwiseSeparableConv3d, self).__init__()
        self.depthwise = nn.Conv3d(in_channels, in_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=in_channels, bias=False)
        self.pointwise = nn.Conv3d(in_channels, out_channels, 1, bias=False)
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class GatedAttentionUnit(nn.Module):
    def __init__(self, in_c, out_c, kernel_size):
        super(GatedAttentionUnit, self).__init__()
        self.w1 = nn.Sequential(
            DepthwiseSeparableConv3d(in_c, out_c, kernel_size, padding=kernel_size//2),
            nn.Sigmoid()
        )
        self.w2 = nn.Sequential(
            DepthwiseSeparableConv3d(in_c, out_c, kernel_size + 2, padding=(kernel_size + 2)//2),
            nn.GELU()
        )
        self.wo = nn.Sequential(
            DepthwiseSeparableConv3d(out_c, out_c, kernel_size, padding=kernel_size//2),
            nn.GELU()
        )
        self.cw = nn.Conv3d(in_c, out_c, 1)
    def forward(self, x):
        x1, x2 = self.w1(x), self.w2(x)
        # 对齐空间尺寸（如有必要）
        if x1.shape != x2.shape:
            x2 = F.interpolate(x2, size=x1.shape[2:], mode='nearest')
        wo = self.wo(x1 * x2)
        out = wo + self.cw(x)
        return out


class DGABlock(nn.Module):
    def __init__(self, in_c, out_c, k_size=3, dilated_ratio=[7, 5, 2, 1]):
        super(DGABlock, self).__init__()
        # Split Dilated Conv Unit
        self.mda0 = DepthwiseSeparableConv3d(in_c//4, in_c//4, k_size, padding=(k_size+(k_size-1)*(dilated_ratio[0]-1))//2, dilation=dilated_ratio[0])
        self.mda1 = DepthwiseSeparableConv3d(in_c//4, in_c//4, k_size, padding=(k_size+(k_size-1)*(dilated_ratio[1]-1))//2, dilation=dilated_ratio[1])
        self.mda2 = DepthwiseSeparableConv3d(in_c//4, in_c//4, k_size, padding=(k_size+(k_size-1)*(dilated_ratio[2]-1))//2, dilation=dilated_ratio[2])
        self.mda3 = DepthwiseSeparableConv3d(in_c//4, in_c//4, k_size, padding=(k_size+(k_size-1)*(dilated_ratio[3]-1))//2, dilation=dilated_ratio[3])
        self.norm_layer = nn.GroupNorm(4, in_c)
        self.conv = nn.Conv3d(in_c, in_c, 1)
        # Gated Attention Unit
        self.gau = GatedAttentionUnit(in_c, out_c, 3)
    def forward(self, x):
        # Split
        x_split = torch.chunk(x, 4, dim=1)
        x0 = self.mda0(x_split[0])
        x1 = self.mda1(x_split[1])
        x2 = self.mda2(x_split[2])
        x3 = self.mda3(x_split[3])
        # Concat
        x_cat = torch.cat((x0, x1, x2, x3), dim=1)
        x_cat = self.norm_layer(self.conv(x_cat))
        # Gated Attention
        out = self.gau(x_cat)
        return out


class EncoderBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = nn.Conv3d(in_c, out_c, 3, padding=1)
        self.bn = nn.BatchNorm3d(out_c)
        self.relu = nn.ReLU(inplace=True)
        self.iea = IEAblock(out_c)
        self.dga = DGABlock(out_c, out_c)
    def forward(self, x):
        x = self.relu(self.bn(self.conv(x)))
        x = self.iea(x)
        x = self.dga(x)
        return x



# ========== 3D注意力桥模块 ===========
class Channel_Att_Bridge3D(nn.Module):
    def __init__(self, c_list, split_att='fc'):
        super().__init__()
        c_list_sum = sum(c_list) - c_list[-1]
        self.split_att = split_att
        self.avgpool = nn.AdaptiveAvgPool3d(1)
        self.get_all_att = nn.Conv1d(1, 1, kernel_size=3, padding=1, bias=False)
        self.att1 = nn.Linear(c_list_sum, c_list[0]) if split_att == 'fc' else nn.Conv1d(c_list_sum, c_list[0], 1)
        self.att2 = nn.Linear(c_list_sum, c_list[1]) if split_att == 'fc' else nn.Conv1d(c_list_sum, c_list[1], 1)
        self.att3 = nn.Linear(c_list_sum, c_list[2]) if split_att == 'fc' else nn.Conv1d(c_list_sum, c_list[2], 1)
        self.att4 = nn.Linear(c_list_sum, c_list[3]) if split_att == 'fc' else nn.Conv1d(c_list_sum, c_list[3], 1)
        self.att5 = nn.Linear(c_list_sum, c_list[4]) if split_att == 'fc' else nn.Conv1d(c_list_sum, c_list[4], 1)
        self.sigmoid = nn.Sigmoid()
    def forward(self, t1, t2, t3, t4, t5):
        att = torch.cat((self.avgpool(t1), 
                         self.avgpool(t2), 
                         self.avgpool(t3), 
                         self.avgpool(t4), 
                         self.avgpool(t5)), dim=1)
        att = self.get_all_att(att.flatten(2).transpose(1, 2))
        if self.split_att != 'fc':
            att = att.transpose(-1, -2)
        att1 = self.sigmoid(self.att1(att))
        att2 = self.sigmoid(self.att2(att))
        att3 = self.sigmoid(self.att3(att))
        att4 = self.sigmoid(self.att4(att))
        att5 = self.sigmoid(self.att5(att))
        if self.split_att == 'fc':
            att1 = att1.transpose(-1, -2).unsqueeze(-1).unsqueeze(-1).expand_as(t1)
            att2 = att2.transpose(-1, -2).unsqueeze(-1).unsqueeze(-1).expand_as(t2)
            att3 = att3.transpose(-1, -2).unsqueeze(-1).unsqueeze(-1).expand_as(t3)
            att4 = att4.transpose(-1, -2).unsqueeze(-1).unsqueeze(-1).expand_as(t4)
            att5 = att5.transpose(-1, -2).unsqueeze(-1).unsqueeze(-1).expand_as(t5)
        else:
            att1 = att1.unsqueeze(-1).unsqueeze(-1).expand_as(t1)
            att2 = att2.unsqueeze(-1).unsqueeze(-1).expand_as(t2)
            att3 = att3.unsqueeze(-1).unsqueeze(-1).expand_as(t3)
            att4 = att4.unsqueeze(-1).unsqueeze(-1).expand_as(t4)
            att5 = att5.unsqueeze(-1).unsqueeze(-1).expand_as(t5)
        return att1, att2, att3, att4, att5

class Spatial_Att_Bridge3D(nn.Module):
    def __init__(self):
        super().__init__()
        self.shared_conv3d = nn.Sequential(
            nn.Conv3d(2, 1, 7, stride=1, padding=9, dilation=3),
            nn.Sigmoid()
        )
    def forward(self, t1, t2, t3, t4, t5):
        t_list = [t1, t2, t3, t4, t5]
        att_list = []
        for t in t_list:
            avg_out = torch.mean(t, dim=1, keepdim=True)
            max_out, _ = torch.max(t, dim=1, keepdim=True)
            att = torch.cat([avg_out, max_out], dim=1)
            att = self.shared_conv3d(att)
            att_list.append(att)
        return att_list[0], att_list[1], att_list[2], att_list[3], att_list[4]

class SC_Att_Bridge3D(nn.Module):
    def __init__(self, c_list, split_att='fc'):
        super().__init__()
        self.catt = Channel_Att_Bridge3D(c_list, split_att=split_att)
        self.satt = Spatial_Att_Bridge3D()
    def forward(self, t1, t2, t3, t4, t5):
        r1, r2, r3, r4, r5 = t1, t2, t3, t4, t5
        satt1, satt2, satt3, satt4, satt5 = self.satt(t1, t2, t3, t4, t5)
        t1, t2, t3, t4, t5 = satt1 * t1, satt2 * t2, satt3 * t3, satt4 * t4, satt5 * t5
        r1_, r2_, r3_, r4_, r5_ = t1, t2, t3, t4, t5
        t1, t2, t3, t4, t5 = t1 + r1, t2 + r2, t3 + r3, t4 + r4, t5 + r5
        catt1, catt2, catt3, catt4, catt5 = self.catt(t1, t2, t3, t4, t5)
        t1, t2, t3, t4, t5 = catt1 * t1, catt2 * t2, catt3 * t3, catt4 * t4, catt5 * t5
        return t1 + r1_, t2 + r2_, t3 + r3_, t4 + r4_, t5 + r5_


class MALUNet3D(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, base_channels=8, split_att='fc', bridge=True):
        super().__init__()
        c_list = [base_channels, base_channels*2, base_channels*3, base_channels*4, base_channels*6, base_channels*8]
        self.bridge = bridge
        # 编码器
        self.encoder1 = nn.Sequential(
            nn.Conv3d(in_channels, c_list[0], 3, stride=1, padding=1),
        )
        self.encoder2 = nn.Sequential(
            nn.Conv3d(c_list[0], c_list[1], 3, stride=1, padding=1),
        )
        self.encoder3 = nn.Sequential(
            nn.Conv3d(c_list[1], c_list[2], 3, stride=1, padding=1),
        )
        self.encoder4 = nn.Sequential(
            IEAblock(c_list[2]),
            DGABlock(c_list[2], c_list[3]),
        )
        self.encoder5 = nn.Sequential(
            IEAblock(c_list[3]),
            DGABlock(c_list[3], c_list[4]),
        )
        self.encoder6 = nn.Sequential(
            IEAblock(c_list[4]),
            DGABlock(c_list[4], c_list[5]),
        )
        if bridge:
            self.scab = SC_Att_Bridge3D(c_list, split_att)
        self.ebn1 = nn.GroupNorm(4, c_list[0])
        self.ebn2 = nn.GroupNorm(4, c_list[1])
        self.ebn3 = nn.GroupNorm(4, c_list[2])
        self.ebn4 = nn.GroupNorm(4, c_list[3])
        self.ebn5 = nn.GroupNorm(4, c_list[4])
        self.dbn1 = nn.GroupNorm(4, c_list[4])
        self.dbn2 = nn.GroupNorm(4, c_list[3])
        self.dbn3 = nn.GroupNorm(4, c_list[2])
        self.dbn4 = nn.GroupNorm(4, c_list[1])
        self.dbn5 = nn.GroupNorm(4, c_list[0])
        # 解码器
        self.decoder1 = nn.Sequential(
            DGABlock(c_list[5], c_list[4]),
            IEAblock(c_list[4]),
        )
        self.decoder2 = nn.Sequential(
            DGABlock(c_list[4], c_list[3]),
            IEAblock(c_list[3]),
        )
        self.decoder3 = nn.Sequential(
            DGABlock(c_list[3], c_list[2]),
            IEAblock(c_list[2]),
        )
        self.decoder4 = nn.Sequential(
            nn.Conv3d(c_list[2], c_list[1], 3, stride=1, padding=1),
        )
        self.decoder5 = nn.Sequential(
            nn.Conv3d(c_list[1], c_list[0], 3, stride=1, padding=1),
        )
        self.final = nn.Conv3d(c_list[0], out_channels, kernel_size=1)

    def forward(self, x):
        out = F.gelu(F.max_pool3d(self.ebn1(self.encoder1(x)), kernel_size=2, stride=2))
        t1 = out  # b, c0, D/2, H/2, W/2
        out = F.gelu(F.max_pool3d(self.ebn2(self.encoder2(out)), kernel_size=2, stride=2))
        t2 = out  # b, c1, D/4, H/4, W/4
        out = F.gelu(F.max_pool3d(self.ebn3(self.encoder3(out)), kernel_size=2, stride=2))
        t3 = out  # b, c2, D/8, H/8, W/8
        out = F.gelu(F.max_pool3d(self.ebn4(self.encoder4(out)), kernel_size=2, stride=2))
        t4 = out  # b, c3, D/16, H/16, W/16
        out = F.gelu(F.max_pool3d(self.ebn5(self.encoder5(out)), kernel_size=2, stride=2))
        t5 = out  # b, c4, D/32, H/32, W/32
        if self.bridge:
            t1, t2, t3, t4, t5 = self.scab(t1, t2, t3, t4, t5)
        out = F.gelu(self.encoder6(out))  # b, c5, D/32, H/32, W/32
        out5 = F.gelu(self.dbn1(self.decoder1(out)))
        if out5.shape != t5.shape:
            out5 = F.interpolate(out5, size=t5.shape[2:], mode='nearest')
        out5 = torch.add(out5, t5)
        out4 = F.gelu(F.interpolate(self.dbn2(self.decoder2(out5)), scale_factor=2, mode='nearest'))
        if out4.shape != t4.shape:
            out4 = F.interpolate(out4, size=t4.shape[2:], mode='nearest')
        out4 = torch.add(out4, t4)
        out3 = F.gelu(F.interpolate(self.dbn3(self.decoder3(out4)), scale_factor=2, mode='nearest'))
        if out3.shape != t3.shape:
            out3 = F.interpolate(out3, size=t3.shape[2:], mode='nearest')
        out3 = torch.add(out3, t3)
        out2 = F.gelu(F.interpolate(self.dbn4(self.decoder4(out3)), scale_factor=2, mode='nearest'))
        if out2.shape != t2.shape:
            out2 = F.interpolate(out2, size=t2.shape[2:], mode='nearest')
        out2 = torch.add(out2, t2)
        out1 = F.gelu(F.interpolate(self.dbn5(self.decoder5(out2)), scale_factor=2, mode='nearest'))
        if out1.shape != t1.shape:
            out1 = F.interpolate(out1, size=t1.shape[2:], mode='nearest')
        out1 = torch.add(out1, t1)
        out0 = F.interpolate(self.final(out1), scale_factor=2, mode='nearest')
        return out0


if __name__ == '__main__':
    images = torch.randn(1, 1, 112, 112, 112)
    model = MALUNet3D(in_channels=1, out_channels=1, base_channels=8)
    y = model(images)
    print(f"输入尺寸: {images.shape}")
    print(f"输出尺寸: {y.shape}")
