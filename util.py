"""
util.py
"""
import os
import numpy as np
import torch
from PIL import Image

def mkdirs(paths):
    """如果文件夹不存在，则创建（支持列表或单个路径）"""
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)

def mkdir(path):
    """创建单个文件夹"""
    if not os.path.exists(path):
        try:
            os.makedirs(path)
        except OSError:
            pass

def tensor2im(input_image, imtype=np.uint8):
    """
    将 Tensor 转换为 Numpy 图片 (用于可视化)
    [-1, 1] -> [0, 255]
    """
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):
            image_tensor = input_image.data
        else:
            return input_image
        image_numpy = image_tensor[0].cpu().float().numpy()
        if image_numpy.shape[0] == 1:
            image_numpy = np.tile(image_numpy, (3, 1, 1))
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    else:
        image_numpy = input_image
    return image_numpy.astype(imtype)

def save_image(image_numpy, image_path):
    """保存 numpy 图片到磁盘"""
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)
    
# -----------------------------------------------------------------------------
# [新增] 3D NIfTI 保存功能
# -----------------------------------------------------------------------------
def save_nii(tensor, save_path):
    """
    将 Tensor 保存为 .nii 文件
    Tensor Shape: (B, C, D, H, W) -> NII Shape: (D, H, W) or (H, W, D)
    """
    if isinstance(tensor, torch.Tensor):
        # 取 Batch 0, Channel 0 -> (D, H, W)
        data = tensor[0, 0, ...].cpu().float().numpy()
    else:
        data = tensor
        
    # 反归一化: [-1, 1] -> [-60, 0] (假设原始范围大概是这个)
    # 或者为了可视化方便，直接存归一化后的值也可以，这里我们存原始值
    # data = (data + 1.0) / 2.0 * 60.0 - 60.0 
    
    # 简单的转置以适配常见软件视角 (视情况而定，通常不需要太复杂)
    # nibabel 默认是 (x, y, z)，pytorch 是 (z, y, x)
    data = data.transpose(2, 1, 0) 
    
    img = nib.Nifti1Image(data, np.eye(4))
    nib.save(img, save_path)