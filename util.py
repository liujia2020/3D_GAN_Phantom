"""
util.py - 最终修正版 (纯净 Save)
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
# 滑窗推理工具
# ==============================================================================
def get_hann_weight(patch_size):
    """
    生成 3D 汉宁窗 (Hann Window)。
    在 50% 重叠步长下，相邻的汉宁窗相加完美等于 1.0，可彻底消除拼接缝隙。
    """
    d, h, w = patch_size
    
    # 使用 numpy 生成三个方向的 1D 汉宁窗
    window_d = np.hanning(d)
    window_h = np.hanning(h)
    window_w = np.hanning(w)
    
    # 利用 numpy 的广播机制，将 1D 窗扩展成 3D 权重矩阵
    weight_3d = window_d[:, None, None] * window_h[None, :, None] * window_w[None, None, :]
    
    # 加上一个极小值 1e-5，防止边缘刚好全是 0 导致后续除零报错
    weight_3d = np.clip(weight_3d, 1e-5, 1.0)
    
    return weight_3d.astype(np.float32)

def predict_sliding_window(model, input_vol, patch_size=(128, 64, 64), stride=(64, 32, 32)):
    # 1. 预处理 [-60, 0] -> [-1, 1]
    norm_min, norm_max = -60.0, 0.0
    input_vol = np.clip(input_vol, norm_min, norm_max)
    img_norm = (input_vol - norm_min) / (norm_max - norm_min) * 2.0 - 1.0
    
    D, H, W = img_norm.shape
    pd, ph, pw = patch_size
    sd, sh, sw = stride
    
    # 2. Padding
    pad_d = (pd - D % pd) % pd
    pad_h = (ph - H % ph) % ph
    pad_w = (pw - W % pw) % pw
    img_pad = np.pad(img_norm, ((0, pad_d+pd), (0, pad_h+ph), (0, pad_w+pw)), mode='reflect')
    
    # 3. 初始化
    output_vol = np.zeros_like(img_pad)
    weight_map = np.zeros_like(img_pad)
    patch_weight = get_hann_weight(patch_size)
    
    model.eval()
    
    # 4. 滑窗
    z_steps = list(range(0, img_pad.shape[0] - pd + 1, sd))
    y_steps = list(range(0, img_pad.shape[1] - ph + 1, sh))
    x_steps = list(range(0, img_pad.shape[2] - pw + 1, sw))
    
    with torch.no_grad():
        for z in z_steps:
            for y in y_steps:
                for x in x_steps:
                    patch = img_pad[z:z+pd, y:y+ph, x:x+pw]
                    patch_tensor = torch.from_numpy(patch).unsqueeze(0).unsqueeze(0).float().cuda()
                    fake_patch = model.netG(patch_tensor)
                    fake_patch = fake_patch.squeeze().cpu().numpy()
                    
                    output_vol[z:z+pd, y:y+ph, x:x+pw] += fake_patch * patch_weight
                    weight_map[z:z+pd, y:y+ph, x:x+pw] += patch_weight
                    
    # 5. 归一化 & 裁剪
    weight_map[weight_map == 0] = 1.0
    output_vol /= weight_map
    output_vol = (output_vol + 1.0) / 2.0 * (norm_max - norm_min) + norm_min
    final_vol = output_vol[:D, :H, :W]
    
    return final_vol

# ==============================================================================
# [修正] 纯净版 save_nii (禁止任何隐式转置)
# ==============================================================================
def save_nii(data, save_path, affine):
    """
    data: numpy array
    save_path: path
    affine: 4x4 matrix
    """
    if isinstance(data, torch.Tensor):
        data = data.cpu().float().numpy()
        
    # 这里的逻辑：此时的 data 应该已经被 test.py 还原回了和 affine 匹配的形状
    # 所以直接保存，不做任何多余的 transpose
    
    img = nib.Nifti1Image(data, affine)
    nib.save(img, save_path)