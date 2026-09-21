import torch
from torch import nn
import torch.nn.functional as F
import math

__all__ = ['UNext3D', 'UNext3D_S']

try:
    from timm.models.layers import DropPath, trunc_normal_
except Exception:
    def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
        if hasattr(nn.init, "trunc_normal_"):
            return nn.init.trunc_normal_(tensor, mean=mean, std=std, a=a, b=b)
        with torch.no_grad():
            tensor.normal_(mean=mean, std=std)
            tensor.clamp_(min=a, max=b)
            return tensor

    def drop_path(x, drop_prob: float = 0., training: bool = False):
        if drop_prob == 0. or not training:
            return x
        keep_prob = 1 - drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor

    class DropPath(nn.Module):
        def __init__(self, drop_prob: float = 0.):
            super().__init__()
            self.drop_prob = drop_prob

        def forward(self, x):
            return drop_path(x, self.drop_prob, self.training)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv3d:
    """1x1 convolution (3D)"""
    return nn.Conv3d(in_planes, out_planes, kernel_size=1, stride=1, bias=False)


# ====== 原代码里这个 shift() 是无效残留，我保持原样但不使用 ======
def shift(dim):
    x_shift = [torch.roll(x_c, shift, dim) for x_c, shift in zip(xs, range(-self.pad, self.pad + 1))]
    x_cat = torch.cat(x_shift, 1)
    x_cat = torch.narrow(x_cat, 2, self.pad, H)
    x_cat = torch.narrow(x_cat, 3, self.pad, W)
    return x_cat


# ==========================
#      3D Depthwise Conv
# ==========================
class DWConv3D(nn.Module):
    def __init__(self, dim=768):
        super(DWConv3D, self).__init__()
        self.dwconv = nn.Conv3d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, D, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, D, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)
        return x


# ==========================
#        3D shiftmlp
# ==========================
class shiftmlp3D(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.GELU, drop=0., shift_size=5):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.dim = in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv3D(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

        self.shift_size = shift_size
        self.pad = shift_size // 2

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, D, H, W):
        B, N, C = x.shape

        # ===== shift along Depth (dim=2) =====
        xn = x.transpose(1, 2).view(B, C, D, H, W).contiguous()
        xn = F.pad(xn, (self.pad, self.pad, self.pad, self.pad, self.pad, self.pad), "constant", 0)

        xs = torch.chunk(xn, self.shift_size, 1)
        x_shift = [torch.roll(x_c, shift, 2) for x_c, shift in zip(xs, range(-self.pad, self.pad + 1))]
        x_cat = torch.cat(x_shift, 1)

        x_cat = torch.narrow(x_cat, 2, self.pad, D)
        x_cat = torch.narrow(x_cat, 3, self.pad, H)
        x_s = torch.narrow(x_cat, 4, self.pad, W)

        x_s = x_s.reshape(B, C, D * H * W).contiguous()
        x_shift_r = x_s.transpose(1, 2)

        x = self.fc1(x_shift_r)
        x = self.dwconv(x, D, H, W)
        x = self.act(x)
        x = self.drop(x)

        # ===== shift along Width (dim=4) =====
        xn = x.transpose(1, 2).view(B, C, D, H, W).contiguous()
        xn = F.pad(xn, (self.pad, self.pad, self.pad, self.pad, self.pad, self.pad), "constant", 0)

        xs = torch.chunk(xn, self.shift_size, 1)
        x_shift = [torch.roll(x_c, shift, 4) for x_c, shift in zip(xs, range(-self.pad, self.pad + 1))]
        x_cat = torch.cat(x_shift, 1)

        x_cat = torch.narrow(x_cat, 2, self.pad, D)
        x_cat = torch.narrow(x_cat, 3, self.pad, H)
        x_s = torch.narrow(x_cat, 4, self.pad, W)

        x_s = x_s.reshape(B, C, D * H * W).contiguous()
        x_shift_c = x_s.transpose(1, 2)

        x = self.fc2(x_shift_c)
        x = self.drop(x)

        return x


# ==========================
#      shiftedBlock 3D
# ==========================
class shiftedBlock3D(nn.Module):
    def __init__(self, dim, num_heads=1, mlp_ratio=4.,
                 qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1):
        super().__init__()

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)

        self.mlp = shiftmlp3D(in_features=dim, hidden_features=mlp_hidden_dim,
                             act_layer=act_layer, drop=drop)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, D, H, W):
        x = x + self.drop_path(self.mlp(self.norm2(x), D, H, W))
        return x


# ==========================
#      OverlapPatchEmbed 3D
# ==========================
class OverlapPatchEmbed3D(nn.Module):
    """ Volume to Patch Embedding """

    def __init__(self, img_size=112, patch_size=3, stride=2, in_chans=64, embed_dim=128):
        super().__init__()

        self.img_size = img_size
        self.patch_size = patch_size
        self.D = img_size // patch_size
        self.H = img_size // patch_size
        self.W = img_size // patch_size

        self.proj = nn.Conv3d(
            in_chans, embed_dim,
            kernel_size=patch_size,
            stride=stride,
            padding=patch_size // 2
        )
        self.norm = nn.LayerNorm(embed_dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv3d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.proj(x)
        _, _, D, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, D, H, W


# ==========================
#         UNext 3D
# ==========================
class UNext3D(nn.Module):

    def __init__(self, num_classes, input_channels=1,
                 deep_supervision=False, img_size=112, patch_size=16, in_chans=1,
                 embed_dims=[128, 160, 256],
                 num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4],
                 qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[1, 1, 1], sr_ratios=[8, 4, 2, 1], **kwargs):
        super().__init__()

        # ===== Encoder (Conv Stage) =====
        self.encoder1 = nn.Conv3d(input_channels, 16, 3, stride=1, padding=1)
        self.encoder2 = nn.Conv3d(16, 32, 3, stride=1, padding=1)
        self.encoder3 = nn.Conv3d(32, 128, 3, stride=1, padding=1)

        self.ebn1 = nn.BatchNorm3d(16)
        self.ebn2 = nn.BatchNorm3d(32)
        self.ebn3 = nn.BatchNorm3d(128)

        self.norm3 = norm_layer(embed_dims[1])
        self.norm4 = norm_layer(embed_dims[2])

        self.dnorm3 = norm_layer(160)
        self.dnorm4 = norm_layer(128)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        # ===== Tokenized MLP blocks =====
        self.block1 = nn.ModuleList([
            shiftedBlock3D(dim=embed_dims[1], num_heads=num_heads[0], mlp_ratio=1,
                           qkv_bias=qkv_bias, qk_scale=qk_scale,
                           drop=drop_rate, attn_drop=attn_drop_rate,
                           drop_path=dpr[0], norm_layer=norm_layer, sr_ratio=sr_ratios[0])
        ])

        self.block2 = nn.ModuleList([
            shiftedBlock3D(dim=embed_dims[2], num_heads=num_heads[0], mlp_ratio=1,
                           qkv_bias=qkv_bias, qk_scale=qk_scale,
                           drop=drop_rate, attn_drop=attn_drop_rate,
                           drop_path=dpr[1], norm_layer=norm_layer, sr_ratio=sr_ratios[0])
        ])

        self.dblock1 = nn.ModuleList([
            shiftedBlock3D(dim=embed_dims[1], num_heads=num_heads[0], mlp_ratio=1,
                           qkv_bias=qkv_bias, qk_scale=qk_scale,
                           drop=drop_rate, attn_drop=attn_drop_rate,
                           drop_path=dpr[0], norm_layer=norm_layer, sr_ratio=sr_ratios[0])
        ])

        self.dblock2 = nn.ModuleList([
            shiftedBlock3D(dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=1,
                           qkv_bias=qkv_bias, qk_scale=qk_scale,
                           drop=drop_rate, attn_drop=attn_drop_rate,
                           drop_path=dpr[1], norm_layer=norm_layer, sr_ratio=sr_ratios[0])
        ])

        # ===== Patch Embedding =====
        self.patch_embed3 = OverlapPatchEmbed3D(
            img_size=img_size // 4, patch_size=3, stride=2,
            in_chans=embed_dims[0], embed_dim=embed_dims[1]
        )

        self.patch_embed4 = OverlapPatchEmbed3D(
            img_size=img_size // 8, patch_size=3, stride=2,
            in_chans=embed_dims[1], embed_dim=embed_dims[2]
        )

        # ===== Decoder =====
        self.decoder1 = nn.Conv3d(256, 160, 3, stride=1, padding=1)
        self.decoder2 = nn.Conv3d(160, 128, 3, stride=1, padding=1)
        self.decoder3 = nn.Conv3d(128, 32, 3, stride=1, padding=1)
        self.decoder4 = nn.Conv3d(32, 16, 3, stride=1, padding=1)
        self.decoder5 = nn.Conv3d(16, 16, 3, stride=1, padding=1)

        self.dbn1 = nn.BatchNorm3d(160)
        self.dbn2 = nn.BatchNorm3d(128)
        self.dbn3 = nn.BatchNorm3d(32)
        self.dbn4 = nn.BatchNorm3d(16)

        self.final = nn.Conv3d(16, num_classes, kernel_size=1)
        self.soft = nn.Softmax(dim=1)

    def forward(self, x):
        B = x.shape[0]

        # ===== Encoder =====
        out = F.relu(F.max_pool3d(self.ebn1(self.encoder1(x)), 2, 2))
        t1 = out

        out = F.relu(F.max_pool3d(self.ebn2(self.encoder2(out)), 2, 2))
        t2 = out

        out = F.relu(F.max_pool3d(self.ebn3(self.encoder3(out)), 2, 2))
        t3 = out

        # ===== Stage 4 (tokenized mlp) =====
        out, D, H, W = self.patch_embed3(out)
        for blk in self.block1:
            out = blk(out, D, H, W)
        out = self.norm3(out)
        out = out.reshape(B, D, H, W, -1).permute(0, 4, 1, 2, 3).contiguous()
        t4 = out

        # ===== Bottleneck =====
        out, D, H, W = self.patch_embed4(out)
        for blk in self.block2:
            out = blk(out, D, H, W)
        out = self.norm4(out)
        out = out.reshape(B, D, H, W, -1).permute(0, 4, 1, 2, 3).contiguous()

        # ===== Decoder Stage 4 =====
        out = F.relu(F.interpolate(self.dbn1(self.decoder1(out)),
                                   size=t4.shape[2:], mode='trilinear', align_corners=False))
        out = torch.add(out, t4)

        _, _, D, H, W = out.shape
        out = out.flatten(2).transpose(1, 2)
        for blk in self.dblock1:
            out = blk(out, D, H, W)

        # ===== Decoder Stage 3 =====
        out = self.dnorm3(out)
        out = out.reshape(B, D, H, W, -1).permute(0, 4, 1, 2, 3).contiguous()

        out = F.relu(F.interpolate(self.dbn2(self.decoder2(out)),
                                   size=t3.shape[2:], mode='trilinear', align_corners=False))
        out = torch.add(out, t3)

        _, _, D, H, W = out.shape
        out = out.flatten(2).transpose(1, 2)
        for blk in self.dblock2:
            out = blk(out, D, H, W)

        out = self.dnorm4(out)
        out = out.reshape(B, D, H, W, -1).permute(0, 4, 1, 2, 3).contiguous()

        out = F.relu(F.interpolate(self.dbn3(self.decoder3(out)),
                                   size=t2.shape[2:], mode='trilinear', align_corners=False))
        out = torch.add(out, t2)

        out = F.relu(F.interpolate(self.dbn4(self.decoder4(out)),
                                   size=t1.shape[2:], mode='trilinear', align_corners=False))
        out = torch.add(out, t1)

        out = F.relu(F.interpolate(self.decoder5(out),
                                   size=x.shape[2:], mode='trilinear', align_corners=False))

        return self.final(out)


# ==========================
#        UNext_S 3D
# ==========================
class UNext3D_S(nn.Module):

    def __init__(self, num_classes, input_channels=1,
                 deep_supervision=False, img_size=112, patch_size=16, in_chans=1,
                 embed_dims=[32, 64, 128, 512],
                 num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4],
                 qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[1, 1, 1], sr_ratios=[8, 4, 2, 1], **kwargs):
        super().__init__()

        self.encoder1 = nn.Conv3d(input_channels, 8, 3, stride=1, padding=1)
        self.encoder2 = nn.Conv3d(8, 16, 3, stride=1, padding=1)
        self.encoder3 = nn.Conv3d(16, 32, 3, stride=1, padding=1)

        self.ebn1 = nn.BatchNorm3d(8)
        self.ebn2 = nn.BatchNorm3d(16)
        self.ebn3 = nn.BatchNorm3d(32)

        self.norm3 = norm_layer(embed_dims[1])
        self.norm4 = norm_layer(embed_dims[2])

        self.dnorm3 = norm_layer(64)
        self.dnorm4 = norm_layer(32)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        self.block1 = nn.ModuleList([
            shiftedBlock3D(dim=embed_dims[1], num_heads=num_heads[0], mlp_ratio=1,
                           qkv_bias=qkv_bias, qk_scale=qk_scale,
                           drop=drop_rate, attn_drop=attn_drop_rate,
                           drop_path=dpr[0], norm_layer=norm_layer, sr_ratio=sr_ratios[0])
        ])

        self.block2 = nn.ModuleList([
            shiftedBlock3D(dim=embed_dims[2], num_heads=num_heads[0], mlp_ratio=1,
                           qkv_bias=qkv_bias, qk_scale=qk_scale,
                           drop=drop_rate, attn_drop=attn_drop_rate,
                           drop_path=dpr[1], norm_layer=norm_layer, sr_ratio=sr_ratios[0])
        ])

        self.dblock1 = nn.ModuleList([
            shiftedBlock3D(dim=embed_dims[1], num_heads=num_heads[0], mlp_ratio=1,
                           qkv_bias=qkv_bias, qk_scale=qk_scale,
                           drop=drop_rate, attn_drop=attn_drop_rate,
                           drop_path=dpr[0], norm_layer=norm_layer, sr_ratio=sr_ratios[0])
        ])

        self.dblock2 = nn.ModuleList([
            shiftedBlock3D(dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=1,
                           qkv_bias=qkv_bias, qk_scale=qk_scale,
                           drop=drop_rate, attn_drop=attn_drop_rate,
                           drop_path=dpr[1], norm_layer=norm_layer, sr_ratio=sr_ratios[0])
        ])

        self.patch_embed3 = OverlapPatchEmbed3D(
            img_size=img_size // 4, patch_size=3, stride=2,
            in_chans=embed_dims[0], embed_dim=embed_dims[1]
        )

        self.patch_embed4 = OverlapPatchEmbed3D(
            img_size=img_size // 8, patch_size=3, stride=2,
            in_chans=embed_dims[1], embed_dim=embed_dims[2]
        )

        self.decoder1 = nn.Conv3d(128, 64, 3, stride=1, padding=1)
        self.decoder2 = nn.Conv3d(64, 32, 3, stride=1, padding=1)
        self.decoder3 = nn.Conv3d(32, 16, 3, stride=1, padding=1)
        self.decoder4 = nn.Conv3d(16, 8, 3, stride=1, padding=1)
        self.decoder5 = nn.Conv3d(8, 8, 3, stride=1, padding=1)

        self.dbn1 = nn.BatchNorm3d(64)
        self.dbn2 = nn.BatchNorm3d(32)
        self.dbn3 = nn.BatchNorm3d(16)
        self.dbn4 = nn.BatchNorm3d(8)

        self.final = nn.Conv3d(8, num_classes, kernel_size=1)
        self.soft = nn.Softmax(dim=1)

    def forward(self, x):
        B = x.shape[0]

        out = F.relu(F.max_pool3d(self.ebn1(self.encoder1(x)), 2, 2))
        t1 = out

        out = F.relu(F.max_pool3d(self.ebn2(self.encoder2(out)), 2, 2))
        t2 = out

        out = F.relu(F.max_pool3d(self.ebn3(self.encoder3(out)), 2, 2))
        t3 = out

        out, D, H, W = self.patch_embed3(out)
        for blk in self.block1:
            out = blk(out, D, H, W)
        out = self.norm3(out)
        out = out.reshape(B, D, H, W, -1).permute(0, 4, 1, 2, 3).contiguous()
        t4 = out

        out, D, H, W = self.patch_embed4(out)
        for blk in self.block2:
            out = blk(out, D, H, W)
        out = self.norm4(out)
        out = out.reshape(B, D, H, W, -1).permute(0, 4, 1, 2, 3).contiguous()

        out = F.relu(F.interpolate(self.dbn1(self.decoder1(out)),
                                   size=t4.shape[2:], mode='trilinear', align_corners=False))
        out = torch.add(out, t4)

        _, _, D, H, W = out.shape
        out = out.flatten(2).transpose(1, 2)
        for blk in self.dblock1:
            out = blk(out, D, H, W)

        out = self.dnorm3(out)
        out = out.reshape(B, D, H, W, -1).permute(0, 4, 1, 2, 3).contiguous()

        out = F.relu(F.interpolate(self.dbn2(self.decoder2(out)),
                                   size=t3.shape[2:], mode='trilinear', align_corners=False))
        out = torch.add(out, t3)

        _, _, D, H, W = out.shape
        out = out.flatten(2).transpose(1, 2)
        for blk in self.dblock2:
            out = blk(out, D, H, W)

        out = self.dnorm4(out)
        out = out.reshape(B, D, H, W, -1).permute(0, 4, 1, 2, 3).contiguous()

        out = F.relu(F.interpolate(self.dbn3(self.decoder3(out)),
                                   size=t2.shape[2:], mode='trilinear', align_corners=False))
        out = torch.add(out, t2)

        out = F.relu(F.interpolate(self.dbn4(self.decoder4(out)),
                                   size=t1.shape[2:], mode='trilinear', align_corners=False))
        out = torch.add(out, t1)

        out = F.relu(F.interpolate(self.decoder5(out),
                                   size=x.shape[2:], mode='trilinear', align_corners=False))

        return self.final(out)

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    x = torch.randn(1, 1, 112, 112, 112, device=device)
    model = UNext3D_S(num_classes=2, input_channels=1, img_size=112).to(device)
    with torch.no_grad():
        y = model(x)
    print(y.shape)

# EOF
