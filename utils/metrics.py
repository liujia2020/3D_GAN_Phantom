"""
图像质量评价指标库 (PSNR, SSIM, MAE)
"""
import torch
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr_loss
from skimage.metrics import structural_similarity as ssim_loss
import logging

def tensor2img(tensor):
    """
    将 [-1, 1] 的 Tensor 转换为 [0, 255] 的 numpy uint8 图像。
    用于标准化计算 PSNR/SSIM。
    """
    # (B, C, D, H, W) -> (D, H, W)
    img_np = tensor.squeeze().cpu().detach().numpy()
    
    # 逆归一化: [-1, 1] -> [0, 255]
    img_np = (img_np + 1.0) / 2.0 * 255.0
    
    # 截断并转为整数
    return np.clip(img_np, 0, 255).astype(np.uint8)

def calc_metrics(fake_tensor, real_tensor):
    """
    计算一组图像的 PSNR, SSIM, MAE。
    输入必须是 5D Tensor (B, C, D, H, W) 或 4D Tensor (C, D, H, W)。
    """
    # 1. 转换为标准图像格式 [0, 255]
    fake_img = tensor2img(fake_tensor)
    real_img = tensor2img(real_tensor)
    
    # 2. 计算 MAE (L1 Error) - 在 [0, 255] 尺度上
    # 也可以选择在 [0, 1] 上算，这里为了统一用 255
    mae = np.mean(np.abs(fake_img.astype(np.float32) - real_img.astype(np.float32))) / 255.0

    # 3. 计算 PSNR
    # data_range=255
    psnr = psnr_loss(real_img, fake_img, data_range=255)
    
    # 4. 计算 SSIM (3D)
    # skimage 的 ssim 支持多维，但需要指定 channel_axis=None (因为我们要把整个 3D 体当做一个对象)
    # 对于 3D 超声，我们通常逐切片计算再平均，或者直接算 3D SSIM。
    # 这里采用最严谨的 3D SSIM 计算。
    ssim = ssim_loss(real_img, fake_img, data_range=255, channel_axis=None)
    
    return {'PSNR': psnr, 'SSIM': ssim, 'MAE': mae}