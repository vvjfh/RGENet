import torch
import torch.nn as nn
import torch.nn.functional as F



class Attention(nn.Module):
    def __init__(self,in_channels,K):
        super().__init__()
        self.avgpool=nn.AdaptiveAvgPool3d(1)
        self.net=nn.Conv3d(in_channels, K, kernel_size=1)
        self.sigmoid=nn.Sigmoid()

    def forward(self,x):
        # [N, C, 1, 1, 1]
        att = self.avgpool(x)
        # [N, K, 1, 1, 1]
        att = self.net(att)
        # [N, K]
        att = att.view(x.shape[0], -1)
        att = self.sigmoid(att)
        return att


class CondConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, groups=1, K=4):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.K = K
        self.groups = groups
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.attention = Attention(in_channels=in_channels, K=K)
        self.weight = nn.Parameter(
            torch.randn(K, out_channels, in_channels // groups, kernel_size, kernel_size, kernel_size),
            requires_grad=True
        )
        self.norm = nn.BatchNorm3d(out_channels,eps=1e-6)

    def forward(self,x):
        # attention weights [N, K]
        N, in_channels, D, H, W = x.shape
        softmax_att=self.attention(x)
        # [N, C_in, D, H, W] -> [1, N*C_in, D, H, W]
        x=x.contiguous().view(1, -1, D, H, W)

        # weight [K, C_out, C_in/groups, k, k, k]
        weight = self.weight
        # [K, C_out*(C_in/groups)*k*k*k]
        weight = weight.view(self.K, -1)

        # [N, K] x [K, ...] -> [N, ...]
        aggregate_weight = torch.mm(softmax_att,weight)
        # [N*C_out, C_in/groups, k, k, k]
        aggregate_weight = aggregate_weight.view(
            N*self.out_channels, self.in_channels//self.groups,
            self.kernel_size, self.kernel_size, self.kernel_size)
        # conv3d -> [1, N*C_out, D', H', W']
        output=F.conv3d(x,weight=aggregate_weight,
                        stride=self.stride, padding=self.padding,
                        groups=self.groups*N)
        Dp, Hp, Wp = output.shape[-3:]
        # [N, C_out, D', H', W']
        output=output.view(N, self.out_channels, Dp, Hp, Wp)
        output = self.norm(output)
        return output

def channel_shuffle(x, groups):
    b, c, d, h, w = x.data.size()

    channels_per_group = c // groups

    # reshape
    x = x.view(b, groups, channels_per_group, d, h, w)

    # transpose and flatten back
    x = torch.transpose(x, 1, 2).contiguous()

    x = x.contiguous().view(b, -1, d, h, w)

    return x

class Block(nn.Module):

    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.dwconv = nn.Conv3d(in_channel, in_channel, kernel_size=7, padding=3, groups=in_channel)  # depthwise conv
        self.conv1_1 = nn.Sequential(
            nn.Conv3d(in_channel, 4*in_channel, 1, 1, 0),
            nn.BatchNorm3d(4*in_channel,),
            nn.LeakyReLU()
        )
        self.conv1_2 = nn.Sequential(
            nn.Conv3d(4*in_channel, 1, 1, 1, 0),
            nn.Sigmoid()
            # nn.BatchNorm2d(1),
            # nn.LeakyReLU()
        )
        # self.dw = nn.Sequential(
        #     nn.Conv2d(out_channel, out_channel, 3, stride=1, padding=1, groups=out_channel),
        #     nn.Conv2d(out_channel, out_channel, 1, stride=1, padding=0, groups=1),
        #     nn.BatchNorm2d(out_channel),
        #     nn.LeakyReLU()
        # )
        # self.conv3x3 = nn.Sequential(
        #     nn.Conv2d(in_channel, 2*in_channel, 3, 1, 1),
        #     nn.BatchNorm2d(2*in_channel),
        #     nn.LeakyReLU()
        # )
        self.down = nn.Conv3d(8*in_channel,in_channel,1,1,0)
        self.weight = nn.Conv3d(in_channel, out_channel, kernel_size=3, stride=1, padding=1)


    def forward(self, x):
        input = x
        x = self.dwconv(x)
        a = self.conv1_1(x)
        a2 = self.conv1_2(a)
        b = a * a2
        # b = self.conv3x3(x)

        x = torch.cat([a,b],1)
        x = self.down(x)
        x = x + input
        x =self.weight(x)
        return x


class CeL(nn.Module):
    def __init__(self,in_channel,out_channel):
        super().__init__()


        self.Conv = nn.Sequential(
            nn.Conv3d(in_channel, int(0.5 * in_channel), 3, stride=1, padding=1),
            nn.BatchNorm3d(int(0.5 * in_channel)),
            nn.LeakyReLU()
        )
        self.down1 = nn.Sequential(
            nn.Conv3d(out_channel, 1, 3, stride=1, padding=1),
            nn.BatchNorm3d(1),
            nn.LeakyReLU()
        )
        self.down2 = nn.Sequential(
            nn.ConvTranspose3d(in_channel, out_channel, 2, stride=2, padding=0, output_padding=0),
            nn.BatchNorm3d(out_channel),
            nn.LeakyReLU()
        )


    def forward(self,x,y):
        x = self.down2(x)
        z = x + y
        # z1 = self.Conv(z)
        z2 = self.down1(z)
        z3 = nn.Sigmoid()(z2)
        z4 = torch.mul(z,z3)
        z5 = torch.cat([z,z4],1)
        z6 = self.Conv(z5)
        # x1 = self.De(x)
        return z6

class SelfAttention(nn.Module):
    def __init__(self, in_channel):
        super().__init__()
        self.channel = in_channel
        self.query = nn.Conv3d(in_channel, in_channel // 8, kernel_size=1, stride=1)
        self.key = nn.Conv3d(in_channel, in_channel // 8, kernel_size=1, stride=1)
        self.value = nn.Conv3d(in_channel, in_channel, kernel_size=1, stride=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, input):
        b, c, d, h, w = input.shape
        S = d * h * w
        # q: B, S, C // 8
        q = self.query(input).view(b, -1, S).permute(0, 2, 1)
        # k: B, C // 8, S
        k = self.key(input).view(b, -1, S)
        # v: B, C, S
        v = self.value(input).view(b, -1, S)
        # attn: B, S, S
        attn_matrix = torch.bmm(q, k)
        attn_matrix = self.softmax(attn_matrix)
        out = torch.bmm(v, attn_matrix.permute(0, 2, 1))
        out = out.view(*input.shape)

        return self.gamma * out + input


class Multi_Concat_Block(nn.Module):
    def __init__(self, in_channel,out_channel):
        super().__init__()
        self.dw = nn.Sequential(
            nn.Conv3d(in_channel, in_channel, 3, stride=1, padding=1, groups=in_channel),
            nn.Conv3d(in_channel, in_channel, 1, stride=1, padding=0, groups=1),
            nn.BatchNorm3d(in_channel),
            nn.LeakyReLU()
        )
        self.Conv3 = nn.Sequential(
            nn.Conv3d(in_channel, in_channel, 3,1, 1),
            nn.BatchNorm3d(in_channel),
            nn.LeakyReLU()
        )
        # self.att = SelfAttention(out_channel)
        self.conv = nn.Conv3d(out_channel, out_channel, kernel_size=3, stride=1, padding=1)


    def forward(self, x):

        x1 = self.dw(x)
        x2 = self.Conv3(x)
        x3 = torch.cat([x1,x2],1)
        # x4 = self.conv1x1(x3)
        # x5 = self.att(x3)
        x6 = self.conv(x3)

        return x6

class Multi_Concat_Att(nn.Module):
    def __init__(self, in_channel,out_channel):
        super().__init__()
        self.dw = nn.Sequential(
            nn.Conv3d(in_channel, in_channel, 3, stride=1, padding=1, groups=in_channel),
            nn.Conv3d(in_channel, in_channel, 1, stride=1, padding=0, groups=1),
            nn.BatchNorm3d(in_channel),
            nn.LeakyReLU()
        )
        self.Conv3 = nn.Sequential(
            nn.Conv3d(in_channel, in_channel, 3,1, 1),
            nn.BatchNorm3d(in_channel),
            nn.LeakyReLU()
        )
        self.att = SelfAttention(out_channel)
        self.conv = nn.Conv3d(out_channel, out_channel, kernel_size=3, stride=1, padding=1)


    def forward(self, x):

        x1 = self.dw(x)
        x2 = self.Conv3(x)
        x3 = torch.cat([x1,x2],1)
        # x4 = self.conv1x1(x3)
        x5 = self.att(x3)
        x6 = self.conv(x5)

        return x6
# class De1(nn.Module):
#     def __init__(self,in_channel,out_channel):
#         super().__init__()
#
#         # self.Conv = nn.Sequential(
#         #     nn.Conv2d(in_channel, int(0.5 * in_channel), 3, stride=1, padding=1),
#         #     nn.BatchNorm2d(int(0.5 * in_channel)),
#         #     nn.LeakyReLU()
#         # )
#         self.up = nn.Sequential(
#             nn.ConvTranspose3d(in_channel, int(0.5 * in_channel), 2, stride=2, padding=0, output_padding=0),
#             nn.BatchNorm3d(int(0.5 * in_channel)),
#             nn.LeakyReLU()
#         )
#         self.att = nn.Sequential(
#             nn.Linear(out_channel, out_channel // 8, bias=False),
#             nn.ReLU(inplace=True),
#             nn.Linear(out_channel // 8, out_channel, bias=False),
#             nn.Sigmoid()
#         )
#         self.sigmoid = nn.Sigmoid()
#         self.GAP = nn.AdaptiveAvgPool3d(1)
#         # self.convout = nn.Sequential(
#         #     nn.Conv2d(in_channel,out_channel,3,1,1),
#         #     nn.BatchNorm2d(out_channel),
#         #     nn.LeakyReLU()
#         # )
#         self.weight = CondConv(3*out_channel,out_channel,3,1,1)
#
#     def forward(self,x,y):
#         z = self.up(x)
#         b,c,_, _ = z.size()
#         z2 = self.GAP(z).view(b, c)
#         z3 = self.att(z2).view(b, c, 1, 1)
#         z4 = z * z3
#         z5 = torch.cat([y,z4],1)
#         z6 = self.weight(z5)
#
#         return z6

class De(nn.Module):
    def __init__(self,in_channel,out_channel):
        super().__init__()
        self.up = nn.Sequential(
            # nn.ConvTranspose3d(in_channel, int(0.5 * in_channel), 2, stride=2, padding=0, output_padding=0),
            nn.Conv3d(in_channel,out_channel,1,1,0),
            nn.BatchNorm3d(int(0.5 * in_channel)),
            nn.LeakyReLU()
        )
        self.att = nn.Sequential(
            nn.Linear(out_channel, out_channel // 8, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(out_channel // 8, out_channel, bias=False),
            nn.Sigmoid()
        )
        self.sigmoid = nn.Sigmoid()
        self.GAP = nn.AdaptiveAvgPool3d(1)
        # self.convout = nn.Sequential(
        #     nn.Conv2d(in_channel,out_channel,3,1,1),
        #     nn.BatchNorm2d(out_channel),
        #     nn.LeakyReLU()
        # )
        self.weight = nn.Conv3d(out_channel, out_channel, kernel_size=3, stride=1, padding=1)

    def forward(self,x,y):
        z = self.up(x)
        z = F.interpolate(z,scale_factor=2,mode='trilinear', align_corners=True)
        b,c,_, _, _ = z.size()
        z2 = self.GAP(z).view(b, c)
        z3 = self.att(z2).view(b, c, 1, 1, 1)
        z4 = z * z3
        # z5 = torch.cat([y,z4],1)
        z5 = z4 + y
        z6 = self.weight(z5)

        return z6

class Net(nn.Module):
    def __init__(self, ):
        super().__init__()

        self.layer1 = Block(1, 16)
        self.layer2 = Block(16, 32)
        self.layer3 = Block(32, 64)
        # self.layer4 = Block(128,256)
        # self.layer5 = Block(256, 512)
        self.pool = nn.MaxPool3d(2, 2)
        # self.mul3 = Multi_Concat_Block(64,128)
        self.mul4 = Block(64, 128)
        self.mul5 = Block(128, 256)
        self.attention = SelfAttention(256)
        self.De1 = De(256, 128)
        self.De2 = De(128, 64)
        self.De3 = De(64, 32)
        self.De4 = De(32, 16)
        self.output = nn.Conv3d(16, 1, kernel_size=3, stride=1, padding=1)
        self.conv1_1 = nn.Conv3d(64, 128,1,1,0)
        self.conv1_2 = nn.Conv3d(128, 256,1,1,0)
        self.sigmoid = nn.Sigmoid()
        self.downconv = nn.Conv3d(256, 128, 1, 1, 0)
        # self.ch1 = changechannel(128,256)
        # self.ch2 = changechannel(256,512)



    def forward(self, x):

        x1 = self.layer1(x)
        x1_down = self.pool(x1)
        x2 = self.layer2(x1_down)
        x2_down = self.pool(x2)
        x3 = self.layer3(x2_down)
        x3_down = self.pool(x3)

        a4 = self.mul4(x3_down)
        a4_s = self.sigmoid(a4)
        a4_down = self.pool(a4)

        b4 = self.conv1_1(x3_down)
        b4_s = self.sigmoid(F.interpolate(b4, size=a4_down.size()[2:], mode="trilinear",align_corners=True))

        a4b = a4_down*b4_s
        b4a = a4_s * b4

        b5 = self.conv1_2(b4a)
        a5 = self.mul5(a4b)

        b5_s = self.sigmoid(F.interpolate(b5, size=a5.size()[2:], mode="trilinear", align_corners=True))
        a5_s = self.sigmoid(F.interpolate(a5, size=b5.size()[2:], mode="trilinear",align_corners=True))
        c = a5 *b5_s
        c = self.attention(c)

        cb = b5 * a5_s
        cbdown = self.downconv(cb)
        c1 = self.De1(c,cbdown)
        c2 = self.De2(c1,x3)
        c3 = self.De3(c2,x2)
        c4 = self.De4(c3,x1)
        output = self.output(c4)

        return output




if __name__ == "__main__":
    device = torch.device("cpu")
    x = torch.rand((1, 1, 112, 112, 112), device=device)
    from time import time
    start = time()
    model = Net()
    _ = model(x)
    speed = time() - start
    print("this case use {:.3f} s".format(speed))
