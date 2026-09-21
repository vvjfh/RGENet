# attention_unet3d.py
# PyTorch 3D Attention U-Net (additive attention gate), input size 112x112x112
# Author: you

import torch
import torch.nn as nn
import torch.nn.functional as F


# ------------------------------
# Basic Blocks
# ------------------------------
class ConvBlock3d(nn.Module):
    """(Conv3d -> Norm -> ReLU) * 2"""
    def __init__(self, in_ch, out_ch, norm='instance'):
        super().__init__()
        Norm = nn.InstanceNorm3d if norm == 'instance' else nn.BatchNorm3d
        self.conv1 = nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.norm1 = Norm(out_ch)
        self.conv2 = nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.norm2 = Norm(out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.act(self.norm1(self.conv1(x)))
        x = self.act(self.norm2(self.conv2(x)))
        return x


class UpConv3d(nn.Module):
    """Upsample (trilinear) + 1x1 Conv to reduce channels"""
    def __init__(self, in_ch, out_ch, align_corners=True):
        super().__init__()
        self.align_corners = align_corners
        self.reduce = nn.Conv3d(in_ch, out_ch, kernel_size=1, bias=False)

    def forward(self, x, size_like):
        x = F.interpolate(x, size=size_like, mode='trilinear', align_corners=self.align_corners)
        x = self.reduce(x)
        return x


# ------------------------------
# Additive Attention Gate (3D)
# ------------------------------
class AttentionGate3d(nn.Module):
    """
    Additive attention gate:
        q = W_g * g
        k = W_x * x   (possibly downsampled to match g's spatial size)
        psi = ReLU(q + k) -> psi_conv -> sigmoid
        out = x * psi_upsampled
    Inputs:
        x: skip connection feature (from encoder)
        g: gating signal (from decoder, coarser scale)
    """
    def __init__(self, in_ch_x, in_ch_g, inter_ch=None, use_batchnorm=False):
        super().__init__()
        if inter_ch is None:
            inter_ch = max(in_ch_x // 2, 1)
        Norm = nn.BatchNorm3d if use_batchnorm else nn.InstanceNorm3d

        # Project skip and gate to an intermediate channel dimension
        self.theta_x = nn.Conv3d(in_ch_x, inter_ch, kernel_size=1, bias=False)
        self.phi_g   = nn.Conv3d(in_ch_g, inter_ch, kernel_size=1, bias=False)
        self.norm_x  = Norm(inter_ch)
        self.norm_g  = Norm(inter_ch)

        # Combine + produce attention coefficients
        self.psi     = nn.Conv3d(inter_ch, 1, kernel_size=1, bias=True)
        self.act     = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, g):
        # Project to inter_ch
        theta_x = self.norm_x(self.theta_x(x))
        phi_g   = self.norm_g(self.phi_g(g))

        # Ensure same spatial size (g is usually smaller)
        if theta_x.shape[2:] != phi_g.shape[2:]:
            phi_g = F.interpolate(phi_g, size=theta_x.shape[2:], mode='trilinear', align_corners=True)

        f = self.act(theta_x + phi_g)
        psi = self.sigmoid(self.psi(f))

        # Attention coefficients need to match x's spatial size
        if psi.shape[2:] != x.shape[2:]:
            psi = F.interpolate(psi, size=x.shape[2:], mode='trilinear', align_corners=True)

        return x * psi  # gated skip


# ------------------------------
# Attention U-Net 3D
# ------------------------------
class AttentionUNet3D(nn.Module):
    """
    3D Attention U-Net with 4-level encoder (down to 7^3 for input 112^3).
    Depth: 4 downsamples (112->56->28->14->7)
    """
    def __init__(
        self,
        in_channels=1,
        num_classes=1,
        base_channels=32,
        norm='instance',
        final_act=None,  # 'sigmoid' | 'softmax' | None (logits)
        align_corners=True
    ):
        super().__init__()
        self.final_act = final_act
        self.align_corners = align_corners

        c1 = base_channels            # 16
        c2 = base_channels * 2        # 32
        c3 = base_channels * 4        # 64
        c4 = base_channels * 8        # 128
        c5 = base_channels * 16       # 256  (bottom)

        # Encoder
        self.enc1 = ConvBlock3d(in_channels, c1, norm=norm)     # 112
        self.down1 = nn.MaxPool3d(kernel_size=2, stride=2)      # -> 56

        self.enc2 = ConvBlock3d(c1, c2, norm=norm)              # 56
        self.down2 = nn.MaxPool3d(kernel_size=2, stride=2)      # -> 28

        self.enc3 = ConvBlock3d(c2, c3, norm=norm)              # 28
        self.down3 = nn.MaxPool3d(kernel_size=2, stride=2)      # -> 14

        self.enc4 = ConvBlock3d(c3, c4, norm=norm)              # 14
        self.down4 = nn.MaxPool3d(kernel_size=2, stride=2)      # -> 7

        self.bottom = ConvBlock3d(c4, c5, norm=norm)            # 7

        # Decoder + Attention gates
        self.up4 = UpConv3d(c5, c4, align_corners=align_corners)
        self.att4 = AttentionGate3d(in_ch_x=c4, in_ch_g=c4, inter_ch=c4 // 2)
        self.dec4 = ConvBlock3d(c4 + c4, c4, norm=norm)         # concat(gated skip, up)

        self.up3 = UpConv3d(c4, c3, align_corners=align_corners)
        self.att3 = AttentionGate3d(in_ch_x=c3, in_ch_g=c3, inter_ch=c3 // 2)
        self.dec3 = ConvBlock3d(c3 + c3, c3, norm=norm)

        self.up2 = UpConv3d(c3, c2, align_corners=align_corners)
        self.att2 = AttentionGate3d(in_ch_x=c2, in_ch_g=c2, inter_ch=c2 // 2)
        self.dec2 = ConvBlock3d(c2 + c2, c2, norm=norm)

        self.up1 = UpConv3d(c2, c1, align_corners=align_corners)
        self.att1 = AttentionGate3d(in_ch_x=c1, in_ch_g=c1, inter_ch=c1 // 2)
        self.dec1 = ConvBlock3d(c1 + c1, c1, norm=norm)

        # Final classifier
        self.head = nn.Conv3d(c1, num_classes, kernel_size=1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv3d, nn.ConvTranspose3d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if getattr(m, 'bias', None) is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm3d, nn.InstanceNorm3d, nn.GroupNorm)):
                if getattr(m, 'weight', None) is not None:
                    nn.init.ones_(m.weight)
                if getattr(m, 'bias', None) is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)             # [B, c1, 112,112,112]
        p1 = self.down1(e1)           # [B, c1, 56,56,56]

        e2 = self.enc2(p1)            # [B, c2, 56,56,56]
        p2 = self.down2(e2)           # [B, c2, 28,28,28]

        e3 = self.enc3(p2)            # [B, c3, 28,28,28]
        p3 = self.down3(e3)           # [B, c3, 14,14,14]

        e4 = self.enc4(p3)            # [B, c4, 14,14,14]
        p4 = self.down4(e4)           # [B, c4, 7,7,7]

        btm = self.bottom(p4)         # [B, c5, 7,7,7]

        # Decoder with attention-gated skips
        u4 = self.up4(btm, e4.shape[2:])  # up to 14^3 -> [B, c4, 14,14,14]
        g4 = self.att4(e4, u4)            # gate the skip e4 with u4
        d4 = self.dec4(torch.cat([u4, g4], dim=1))

        u3 = self.up3(d4, e3.shape[2:])   # -> 28^3
        g3 = self.att3(e3, u3)
        d3 = self.dec3(torch.cat([u3, g3], dim=1))

        u2 = self.up2(d3, e2.shape[2:])   # -> 56^3
        g2 = self.att2(e2, u2)
        d2 = self.dec2(torch.cat([u2, g2], dim=1))

        u1 = self.up1(d2, e1.shape[2:])   # -> 112^3
        g1 = self.att1(e1, u1)
        d1 = self.dec1(torch.cat([u1, g1], dim=1))

        logits = self.head(d1)            # [B, num_classes, 112,112,112]

        if self.final_act is None:
            return logits
        elif self.final_act == 'sigmoid':
            return torch.sigmoid(logits)
        elif self.final_act == 'softmax':
            return F.softmax(logits, dim=1)
        else:
            raise ValueError("final_act must be None | 'sigmoid' | 'softmax'")


# ------------------------------
# Quick Sanity Check
# ------------------------------
if __name__ == "__main__":
    # Input size 112^3 as requested
    x = torch.randn(1, 1, 112, 112, 112)
    net = AttentionUNet3D(in_channels=1, num_classes=1, base_channels=16, final_act=None)
    with torch.no_grad():
        y = net(x)
    print("Input :", x.shape)
    print("Output:", y.shape)  # expect [1, 1, 112, 112, 112]
