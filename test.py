"""
test.py - 专业版
"""
import os
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg') # 无窗口模式
import matplotlib.pyplot as plt
from options.test_options import TestOptions
from data import create_dataset
import util

# =========================================================================
# 辅助函数：计算 PSNR
# =========================================================================
def calculate_psnr(img1, img2):
    mse = np.mean((img1.astype(np.float32) - img2.astype(np.float32)) ** 2)
    if mse == 0: return 100
    return 20 * np.log10(255.0 / np.sqrt(mse))

# =========================================================================
# 辅助函数：Tensor转图 (Matplotlib用)
# =========================================================================
def tensor2im_custom(input_image, imtype=np.uint8):
    image_tensor = input_image.data
    image_numpy = image_tensor[0].cpu().float().numpy()
    
    # 取中间切片用于 PNG 展示
    if image_numpy.ndim == 4:
        mid_slice = image_numpy.shape[1] // 2
        image_numpy = image_numpy[:, mid_slice, :, :]
        
    if image_numpy.shape[0] == 1:
        image_numpy = np.tile(image_numpy, (3, 1, 1))
        
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    image_numpy = np.clip(image_numpy, 0, 255).astype(imtype)
    return image_numpy

# =========================================================================
# 绘制论文级对比图 (Input | Fake | Truth)
# =========================================================================
def save_paper_fig(save_path, case_name, model_name, psnr_val, img_lq, img_fake, img_sq):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6.5))
    
    header_txt = f"Model: {model_name}  |  Case: {case_name}  |  PSNR: {psnr_val:.2f} dB"
    fig.suptitle(header_txt, fontsize=20, fontweight='bold', y=0.92)
    
    items = [('Input (LQ)', img_lq), ('Generated (HQ)', img_fake), ('Ground Truth (HQ)', img_sq)]
    
    for ax, (title, img) in zip(axes, items):
        ax.imshow(img)
        ax.set_title(title, fontsize=16, pad=10)
        ax.axis('off')
    
    plt.subplots_adjust(top=0.85, wspace=0.05, left=0.02, right=0.98, bottom=0.02)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

# =========================================================================
# 主测试逻辑
# =========================================================================
if __name__ == '__main__':
    opt = TestOptions().parse()
    opt.num_threads = 0   # 测试时建议单线程，避免 DataLoader 报错
    opt.batch_size = 1    # 测试必须单张跑
    opt.serial_batches = True  # 不打乱顺序
    opt.no_flip = True    # 严禁翻转
    opt.display_id = -1   # 关闭 Visdom

    print("\n" + "="*80)
    print(f"🚀 测试启动: {opt.name}")
    print(f"📂 结果保存路径: {opt.results_dir}")
    print("="*80)

    dataset = create_dataset(opt)
    model = create_model(opt)
    model.setup(opt)
    
    # 结果目录结构: results/Experiment_Name/test_latest/images/
    web_dir = os.path.join(opt.results_dir, opt.name, f'test_{opt.epoch}')
    img_dir = os.path.join(web_dir, 'images')
    if not os.path.exists(img_dir): os.makedirs(img_dir)

    print(f"📊 开始测试 {min(len(dataset), opt.num_test)} 个样本...")

    for i, data in enumerate(dataset):
        if i >= opt.num_test: break
        
        model.set_input(data)
        model.test() # 前向推理
        
        # 获取文件名 (Case Name)
        img_path = model.get_image_paths() # list of paths
        case_name = os.path.basename(img_path[0]) if len(img_path)>0 else f"sample_{i}"
        
        print(f"Processing: {case_name}")

        # 1. 获取 Tensor
        fake_tensor = getattr(model, 'fake_hq', getattr(model, 'fake_B', None))
        real_tensor = getattr(model, 'real_sq', getattr(model, 'real_B', None))
        input_tensor = getattr(model, 'real_lq', getattr(model, 'real_A', None))
        
        if fake_tensor is not None:
            # 2. 生成 PNG 对比图
            img_lq = tensor2im_custom(input_tensor)
            img_fake = tensor2im_custom(fake_tensor)
            img_sq = tensor2im_custom(real_tensor)
            
            psnr = calculate_psnr(img_fake, img_sq)
            
            png_name = f"{case_name}_comparison.png"
            save_paper_fig(os.path.join(img_dir, png_name), case_name, opt.name, psnr, img_lq, img_fake, img_sq)
            
            # 3. 保存 3D NIfTI (Fake Volume)
            nii_name = f"{case_name}_fake.nii.gz"
            save_nii(fake_tensor, os.path.join(img_dir, nii_name))

    print(f"✅ 测试完成！结果已保存在: {img_dir}")