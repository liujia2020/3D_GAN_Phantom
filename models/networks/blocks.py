import torch
import torch.nn as nn
import torch.nn.functional as F

# ==========================================================
# Part 1: Attention Mechanisms (CBAM, etc.) - Pure 2D
# ==========================================================

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
           
        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class CBAM2D(nn.Module):
    def __init__(self, planes, ratio=16, kernel_size=7):
        super(CBAM2D, self).__init__()
        self.ca = ChannelAttention(planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        out = x * self.ca(x)
        result = out * self.sa(out)
        return result

class PixelAwareAttention(nn.Module):
    def __init__(self, in_channels, is_3d=False, temperature=1.0):
        super(PixelAwareAttention, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 1, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.temperature = temperature

    def forward(self, x):
        attn_map = self.sigmoid(self.conv1(x) * self.temperature)
        return x * attn_map

class LocalAwareAttention(nn.Module):
    def __init__(self, F_g, F_l, F_int, is_3d=False):
        super(LocalAwareAttention, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=False)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        if g1.shape[2:] != x1.shape[2:]:
            g1 = F.interpolate(g1, size=x1.shape[2:], mode='bilinear', align_corners=False)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi

# ==========================================================
# Part 2: Building Blocks (ResBlock, ASPP) - Pure 2D
# ==========================================================

class ResnetBlock2D(nn.Module):
    def __init__(self, dim, norm_layer, use_dropout, use_bias, dilation=1):
        super(ResnetBlock2D, self).__init__()
        self.conv_block = self.build_conv_block(dim, norm_layer, use_dropout, use_bias, dilation)
        self.relu = nn.ReLU(False)
        self.cbam = CBAM2D(dim, ratio=16, kernel_size=7)

    def build_conv_block(self, dim, norm_layer, use_dropout, use_bias, dilation):
        conv_block = []
        p = dilation 
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, dilation=dilation, bias=use_bias),
                       norm_layer(dim),
                       nn.ReLU(False)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, dilation=dilation, bias=use_bias),
                       norm_layer(dim)]
        return nn.Sequential(*conv_block)

    def forward(self, x):
        residual = self.conv_block(x)
        residual = self.cbam(residual)
        out = x + residual
        return self.relu(out)

class ASPP2D(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer=nn.BatchNorm2d):
        super(ASPP2D, self).__init__()
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            norm_layer(out_channels),
            nn.LeakyReLU(0.2, True)
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=2, dilation=2, bias=False),
            norm_layer(out_channels),
            nn.LeakyReLU(0.2, True)
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=3, dilation=3, bias=False),
            norm_layer(out_channels),
            nn.LeakyReLU(0.2, True)
        )
        self.branch5_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=True),
            nn.LeakyReLU(0.2, True)
        )
        self.conv_cat = nn.Sequential(
            nn.Conv2d(out_channels * 4, out_channels, kernel_size=1, bias=False),
            norm_layer(out_channels),
            nn.LeakyReLU(0.2, True)
        )

    def forward(self, x):
        [b, c, h, w] = x.size()
        conv1x1 = self.branch1(x)
        conv3x3_d2 = self.branch2(x)
        conv3x3_d3 = self.branch3(x)
        global_feature = F.adaptive_avg_pool2d(x, (1, 1))
        global_feature = self.branch5_conv(global_feature)
        global_feature = F.interpolate(global_feature, size=(h, w), mode='bilinear', align_corners=True)
        feature_cat = torch.cat([conv1x1, conv3x3_d2, conv3x3_d3, global_feature], dim=1)
        return self.conv_cat(feature_cat)