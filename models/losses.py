import torch
import torch.nn as nn
import torchvision.models as models

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