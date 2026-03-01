import torch
import torch.nn as nn
import torchvision.models as models
import math

class GaussianBlurLayer(nn.Module):
    """自定义的高斯模糊层（物理老花镜），用于高低频解耦"""
    def __init__(self, channels=1, kernel_size=7, sigma=3.0):
        super(GaussianBlurLayer, self).__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        self.sigma = sigma

        # 生成 2D 高斯核
        x_coord = torch.arange(kernel_size)
        x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
        y_grid = x_grid.t()
        xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()

        mean = (kernel_size - 1) / 2.
        variance = sigma ** 2.

        gaussian_kernel = (1. / (2. * math.pi * variance)) * torch.exp(
            -torch.sum((xy_grid - mean) ** 2., dim=-1) / (2 * variance)
        )
        gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

        # Reshape 为深度可分离卷积的权重格式
        gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
        gaussian_kernel = gaussian_kernel.repeat(channels, 1, 1, 1)

        self.gaussian_filter = nn.Conv2d(in_channels=channels, out_channels=channels,
                                         kernel_size=kernel_size, groups=channels,
                                         bias=False, padding=kernel_size // 2, padding_mode='reflect')
        self.gaussian_filter.weight.data = gaussian_kernel
        self.gaussian_filter.weight.requires_grad = False # 权重固定，不参与训练

    def forward(self, x):
        return self.gaussian_filter(x)


class CharbonnierLoss(nn.Module):
    """论文公式 (9) 的精确实现：平滑的 L1，绝不会把高频散斑压成塑料块"""
    def __init__(self, eps=1e-6):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        loss = torch.mean(torch.sqrt(diff * diff + self.eps))
        return loss

# =========================================================================
# [模块 1] VGG19 特征提取器
# =========================================================================
class VGG19FeatureExtractor(nn.Module):
    def __init__(self):
        super(VGG19FeatureExtractor, self).__init__()
        vgg19 = models.vgg19(pretrained=True)
        self.features = nn.Sequential(*list(vgg19.features.children())[:30])
        for param in self.features.parameters():
            param.requires_grad = False
        
    def forward(self, x):
        return self.features(x)

# =========================================================================
# [模块 2] Total Variation Loss (TV Loss)
# =========================================================================
class TVLoss(nn.Module):
    def __init__(self, weight=1.0):
        super(TVLoss, self).__init__()
        self.weight = weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self.tensor_size(x[:, :, 1:, :])
        count_w = self.tensor_size(x[:, :, :, 1:])
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        return self.weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size

    def tensor_size(self, t):
        return t.size()[1] * t.size()[2] * t.size()[3]

# =========================================================================
# [模块 3] Frequency Loss (FFL) (已修复为 2.5D 平面输入)
# =========================================================================
class FrequencyLoss(nn.Module):
    def __init__(self):
        super(FrequencyLoss, self).__init__()
        self.criterion = nn.L1Loss()

    def forward(self, pred, target):
        # 降维修复：此时的 pred 和 target 已经是纯净的 4 维张量 (Batch, Channel, Height, Width)
        # 所以完全不需要解包 d(深度)，直接扔进 rfft2 即可，它会自动对最后两个维度 (H, W) 做 2D FFT
        pred_fft = torch.fft.rfft2(pred, norm='ortho')
        target_fft = torch.fft.rfft2(target, norm='ortho')
        
        pred_amp = torch.abs(pred_fft)
        target_amp = torch.abs(target_fft)
        
        return self.criterion(torch.log(pred_amp + 1.0), torch.log(target_amp + 1.0))