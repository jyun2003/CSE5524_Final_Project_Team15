import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict


# ------------------- CBAM -------------------
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        return x * self.sigmoid(avg_out + max_out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        attn = torch.cat([avg_out, max_out], dim=1)
        scale = self.sigmoid(self.conv(attn))
        return x * scale


class CBAM(nn.Module):
    def __init__(self, in_channels, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_channels, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        x = self.ca(x)
        x = self.sa(x)
        return x


# ------------------- Utility -------------------
def conv_layer(in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1, padding=None):
    if padding is None:
        padding = int((kernel_size - 1) / 2) * dilation
    return nn.Conv2d(
        in_channels, out_channels, kernel_size, stride, padding=padding, bias=True, dilation=dilation, groups=groups
    )


def activation(act_type="lrelu", inplace=True, neg_slope=0.05, n_prelu=1):
    act_type = act_type.lower()
    if act_type == "relu":
        return nn.ReLU(inplace)
    elif act_type == "lrelu":
        return nn.LeakyReLU(neg_slope, inplace)
    elif act_type == "prelu":
        return nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    else:
        raise NotImplementedError(f"Activation type [{act_type}] not supported")


def sequential(*args):
    modules = []
    for module in args:
        if isinstance(module, nn.Sequential):
            modules.extend(module.children())
        elif isinstance(module, nn.Module):
            modules.append(module)
    return nn.Sequential(*modules)


# ------------------- ESA -------------------
class ESA(nn.Module):
    def __init__(self, n_feats, conv):
        super(ESA, self).__init__()
        f = n_feats // 4
        self.conv1 = conv(n_feats, f, kernel_size=1)
        self.conv_f = conv(f, f, kernel_size=1)
        self.conv_max = conv(f, f, kernel_size=3, padding=1)
        self.conv2 = conv(f, f, kernel_size=3, stride=2, padding=0)
        self.conv3 = conv(f, f, kernel_size=3, padding=1)
        self.conv3_ = conv(f, f, kernel_size=3, padding=1)
        self.conv4 = conv(f, n_feats, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        c1_ = self.conv1(x)
        c1 = self.conv2(c1_)
        v_max = F.max_pool2d(c1, kernel_size=7, stride=3)
        v_range = self.relu(self.conv_max(v_max))
        c3 = self.relu(self.conv3(v_range))
        c3 = self.conv3_(c3)
        c3 = F.interpolate(c3, size=x.shape[2:], mode="bilinear", align_corners=False)
        cf = self.conv_f(c1_)
        c4 = self.conv4(c3 + cf)
        m = self.sigmoid(c4)
        return x * m


# ------------------- RFDB -------------------
class RFDB(nn.Module):
    def __init__(self, in_channels, distillation_rate=0.25):
        super(RFDB, self).__init__()
        self.dc = in_channels // 2
        self.rc = in_channels

        self.c1_d = conv_layer(in_channels, self.dc, 1)
        self.c1_r = conv_layer(in_channels, self.rc, 3)
        self.c2_d = conv_layer(self.rc, self.dc, 1)
        self.c2_r = conv_layer(self.rc, self.rc, 3)
        self.c3_d = conv_layer(self.rc, self.dc, 1)
        self.c3_r = conv_layer(self.rc, self.rc, 3)
        self.c4 = conv_layer(self.rc, self.dc, 3)

        self.act = activation("lrelu")
        self.c5 = conv_layer(self.dc * 4, in_channels, 1)
        self.esa = ESA(in_channels, conv_layer)

    def forward(self, x):
        d1 = self.act(self.c1_d(x))
        r1 = self.act(self.c1_r(x) + x)

        d2 = self.act(self.c2_d(r1))
        r2 = self.act(self.c2_r(r1) + r1)

        d3 = self.act(self.c3_d(r2))
        r3 = self.act(self.c3_r(r2) + r2)

        r4 = self.act(self.c4(r3))
        out = torch.cat([d1, d2, d3, r4], dim=1)
        out = self.c5(out)
        out = self.esa(out)
        return out


# ------------------- Upsampler -------------------
def pixelshuffle_block(in_channels, out_channels, upscale_factor=2, kernel_size=3, stride=1):
    conv = conv_layer(in_channels, out_channels * (upscale_factor**2), kernel_size, stride)
    return nn.Sequential(conv, nn.PixelShuffle(upscale_factor))


# ------------------- RFDN Model with CBAM -------------------
class RFDN(nn.Module):
    def __init__(self, in_nc=3, nf=50, num_modules=4, out_nc=3, upscale=4):
        super(RFDN, self).__init__()
        self.fea_conv = conv_layer(in_nc, nf, kernel_size=3)

        self.B1 = RFDB(nf)
        self.cbam1 = CBAM(nf)
        self.B2 = RFDB(nf)
        self.cbam2 = CBAM(nf)
        self.B3 = RFDB(nf)
        self.cbam3 = CBAM(nf)
        self.B4 = RFDB(nf)
        self.cbam4 = CBAM(nf)

        self.c = conv_layer(nf * num_modules, nf, kernel_size=1)
        self.act = activation("lrelu")
        self.LR_conv = conv_layer(nf, nf, kernel_size=3)
        self.upsampler = pixelshuffle_block(nf, out_nc, upscale)

    def forward(self, x):
        fea = self.fea_conv(x)

        b1 = self.cbam1(self.B1(fea))
        b2 = self.cbam2(self.B2(b1))
        b3 = self.cbam3(self.B3(b2))
        b4 = self.cbam4(self.B4(b3))

        out_B = self.c(torch.cat([b1, b2, b3, b4], dim=1))
        out_lr = self.LR_conv(self.act(out_B)) + fea
        return self.upsampler(out_lr)
