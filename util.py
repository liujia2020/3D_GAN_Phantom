"""
util.py - 最终修正版 (2.5D 切片推理引擎)
"""
import os
import numpy as np
import torch
from PIL import Image
import nibabel as nib

def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)

def mkdir(path):
    if not os.path.exists(path):
        try:
            os.makedirs(path)
        except OSError:
            pass

def tensor2im(input_image, imtype=np.uint8):
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):
            image_tensor = input_image.data
        else:
            return input_image
        image_numpy = image_tensor[0].cpu().float().numpy()
        if image_numpy.ndim == 4:
            mid_slice = image_numpy.shape[1] // 2
            image_numpy = image_numpy[:, mid_slice, :, :]
        if image_numpy.shape[0] == 1:
            image_numpy = np.tile(image_numpy, (3, 1, 1))
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    else:
        image_numpy = input_image
    return image_numpy.astype(imtype)

def save_image(image_numpy, image_path):
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)

# ==============================================================================
# [核心重构] 2.5D 切片推理工具 (替代笨重的 3D 滑窗)
# ==============================================================================
def predict_slice_by_slice(model, input_vol, opt):
    """
    逐层扫描推断：将 1024 层的 3D 矩阵，切成 1024 次 2.5D 夹心饼干喂给网络。
    不使用任何滑窗和重叠，速度极快，且彻底避免 3D 拉扯伪影。
    """
    # 1. 预处理 [-60, 0] -> [-1, 1]
    norm_min = getattr(opt, 'norm_min', -60.0)
    norm_max = getattr(opt, 'norm_max', 0.0)
    input_vol = np.clip(input_vol, norm_min, norm_max)
    img_norm = (input_vol - norm_min) / (norm_max - norm_min) * 2.0 - 1.0
    
    D, H, W = img_norm.shape
    
    # 初始化一个全空的 3D 矩阵，准备接收网络吐出的高清 2D 切片
    output_vol = np.zeros_like(img_norm)
    
    model.eval()
    with torch.no_grad():
        for z in range(D):
            # 边界保护
            z_prev = max(0, z - 1)
            z_next = min(D - 1, z + 1)
            
            # 抽取 2.5D 夹心饼干
            slice_prev = img_norm[z_prev, :, :]
            slice_curr = img_norm[z, :, :]
            slice_next = img_norm[z_next, :, :]
            
            # 叠成 (3, H, W)
            patch_25d = np.stack([slice_prev, slice_curr, slice_next], axis=0)
            
            # 升维变 Tensor: (1, 3, H, W)
            patch_tensor = torch.from_numpy(patch_25d).unsqueeze(0).float().to(model.device)
            
            # 网络推理，吐出 (1, 1, H, W) 的单层高清图
            fake_slice = model.netG(patch_tensor)
            
            # 降维变 Numpy: (H, W)
            fake_slice_np = fake_slice.squeeze().cpu().numpy()
            
            # 像发扑克牌一样，精确地放回空 3D 矩阵的当前层
            output_vol[z, :, :] = fake_slice_np
            
    # 2. 反归一化 [-1, 1] -> [-60, 0]
    output_vol = (output_vol + 1.0) / 2.0 * (norm_max - norm_min) + norm_min
    
    return output_vol

# ==============================================================================
# 纯净版 save_nii (禁止任何隐式转置)
# ==============================================================================
def save_nii(data, save_path, affine):
    """
    data: numpy array
    save_path: path
    affine: 4x4 matrix
    """
    if isinstance(data, torch.Tensor):
        data = data.cpu().float().numpy()
        
    # 直接保存，不做任何多余的 transpose
    img = nib.Nifti1Image(data, affine)
    nib.save(img, save_path)