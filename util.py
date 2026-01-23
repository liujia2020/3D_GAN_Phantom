import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from skimage.metrics import structural_similarity as ssim_func
from skimage.metrics import peak_signal_noise_ratio as psnr_func

def tensor2im(input_image, imtype=np.float32):
    """
    将 Tensor [-1, 1] 转换为 numpy [0, 1] (线性值)
    用于显示前的反归一化，并自动挤压通道维度以适配 2D 显示和指标计算
    """
    if isinstance(input_image, torch.Tensor):
        image_tensor = input_image.data
    else:
        return input_image
    
    # 这里的 input_image 通常是 [B, C, H, W] (切片后 B=1)
    # 取 Batch 中的第一个样本
    if len(image_tensor.shape) == 4:
        image_tensor = image_tensor[0] # 变成 [C, H, W]
    
    img_np = image_tensor.cpu().float().numpy() 
    
    # [关键修复] 如果是单通道 [1, H, W]，强制挤压成 [H, W]
    # 这样 skimage 的 ssim 才能正确把它识别为 2D 图像，而不是 3D 立体块
    if img_np.ndim == 3 and img_np.shape[0] == 1:
        img_np = np.squeeze(img_np, axis=0)
    
    # [关键] 反归一化: [-1, 1] -> [0, 1]
    # 对应 dataset 中的 (x - 0.5) * 2
    img_np = (img_np + 1.0) / 2.0
    
    # 截断防止溢出 (clip 到 0~1)
    img_np = np.clip(img_np, 0, 1)
    
    return img_np

def log_compression(img_np, dynamic_range=60):
    """
    对 [0, 1] 的线性 numpy 数据进行对数压缩
    """
    # 1. 加上微小值防止 log(0)
    img_log = 20 * np.log10(img_np + 1e-8)
    
    # 2. 动态范围截断
    max_val = np.max(img_log)
    min_val = max_val - dynamic_range
    img_log = np.clip(img_log, min_val, max_val)
    
    # 3. 归一化显示 [0, 1]
    img_show = (img_log - min_val) / dynamic_range
    return img_show

def calc_metrics(img1, img2):
    """
    计算单张图的 PSNR/SSIM
    img1, img2: 范围 [0, 1] 的 numpy 数组 (H, W)
    """
    # data_range=1 表示图像范围是 0~1
    p = psnr_func(img2, img1, data_range=1)
    
    # channel_axis=None 表示这是灰度图(2D)，没有通道维
    s = ssim_func(img2, img1, data_range=1, channel_axis=None) 
    return p, s

def save_training_images(lq, pred, hq, sq, epoch, save_dir, dynamic_range=60):
    """
    保存 5 张对比图: Input | Pred | Diff | HQ | SQ
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # --- 1. 数据预处理 (取中间切片) ---
    # lq: [B, 3, D, H, W] -> 取 depth 中间 -> [B, 3, H, W]
    mid_idx = lq.shape[2] // 2
    
    # Input 融合: 3个角度取平均，模拟传统复合
    lq_slice = torch.mean(lq[:, :, mid_idx, :, :], dim=1, keepdim=True)
    
    pred_slice = pred[:, :, mid_idx, :, :]
    hq_slice = hq[:, :, mid_idx, :, :]
    sq_slice = sq[:, :, mid_idx, :, :]
    
    # --- 2. 转 Numpy [0, 1] (线性反归一化 & Squeeze) ---
    # 只取 Batch 中的第一个样本进行显示
    img_lq = tensor2im(lq_slice[0:1])
    img_pred = tensor2im(pred_slice[0:1])
    img_hq = tensor2im(hq_slice[0:1])
    img_sq = tensor2im(sq_slice[0:1])
    
    # --- 3. 计算监控指标 (Metrics) ---
    # 以 SQ (真值) 为标准
    psnr_pred, ssim_pred = calc_metrics(img_pred, img_sq)
    
    # --- 4. 制作差异图 (Diff Map) ---
    # abs(Pred - SQ)
    diff_map = np.abs(img_pred - img_sq)
    # 差异图通常比较暗，放大 5 倍亮度以便观察误差
    diff_show = np.clip(diff_map * 5.0, 0, 1) 
    
    # --- 5. 对数压缩 (Log Compression) ---
    # 用于解剖结构展示
    show_lq = log_compression(img_lq, dynamic_range)
    show_pred = log_compression(img_pred, dynamic_range)
    show_hq = log_compression(img_hq, dynamic_range)
    show_sq = log_compression(img_sq, dynamic_range)
    
    # --- 6. 画图 (1行5列) ---
    fig, axs = plt.subplots(1, 5, figsize=(22, 5))
    
    # 准备数据和标题
    # 格式: (图片数据, 标题文字, 颜色映射)
    items = [
        (show_lq, f"Input (Mean)\nMin:{img_lq.min():.2f} Max:{img_lq.max():.2f}", 'gray'),
        (show_pred, f"Prediction (AI)\nPSNR:{psnr_pred:.1f} SSIM:{ssim_pred:.2f}\nMin:{img_pred.min():.2f} Max:{img_pred.max():.2f}", 'gray'),
        (diff_show, f"Diff Map (|Pred-SQ|x5)\n(Black=Good, Red=Error)", 'inferno'),
        (show_hq, f"Ref (HQ 33)\nMin:{img_hq.min():.2f} Max:{img_hq.max():.2f}", 'gray'),
        (show_sq, f"GT (SQ 75)\nTarget", 'gray'),
    ]
    
    for i, (img, title, cmap) in enumerate(items):
        # 这里的 vmin/vmax 确保显示范围固定在 0-1
        axs[i].imshow(img, cmap=cmap, vmin=0, vmax=1)
        axs[i].set_title(title, fontsize=11, fontweight='bold')
        axs[i].axis('off')
        
    # 保存
    save_path = os.path.join(save_dir, f"Epoch_{epoch:03d}_Monitor.png")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()