"""
test.py - 2.5D 终极切片扫描版
功能：
1. 彻底抛弃 3D 滑动窗口，改为极速 2D 切片扫描扫描全卷。
2. 强制保留原始 0.036mm 物理分辨率的 Affine 矩阵。
3. 智能处理维度转置与还原，并保存 9 宫格对比图。
"""
import os
import torch
import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from options.test_options import TestOptions
from models import create_model
# 【修改处】引入全新的 predict_slice_by_slice
from util import save_nii, predict_slice_by_slice

def calculate_psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0: return 100
    return 20 * np.log10(255.0 / np.sqrt(mse))

def save_paper_fig_9grid(save_path, case_name, model_name, metrics_dict, vol_lq, vol_fake, vol_sq):
    D, H, W = vol_lq.shape
    idx_z = 500 if D > 500 else D // 2
    idx_x = 64  if W > 64  else W // 2
    idx_y = 64  if H > 64  else H // 2
    
    ax_lq, ax_fk, ax_sq = vol_lq[idx_z,:,:], vol_fake[idx_z,:,:], vol_sq[idx_z,:,:]
    sa_lq, sa_fk, sa_sq = vol_lq[:,:,idx_x], vol_fake[:,:,idx_x], vol_sq[:,:,idx_x]
    co_lq, co_fk, co_sq = vol_lq[:,idx_y,:], vol_fake[:,idx_y,:], vol_sq[:,idx_y,:]
    
    fig, axes = plt.subplots(3, 3, figsize=(15, 18))
    psnr = metrics_dict.get('PSNR', 0)
    fig.suptitle(f"Exp: {model_name} | Case: {case_name} | Vol PSNR: {psnr:.2f} dB", fontsize=22, fontweight='bold', y=0.95)
    
    rows = [("Axial (Z)", [ax_lq, ax_fk, ax_sq]), 
            ("Sagittal (X)", [sa_lq, sa_fk, sa_sq]), 
            ("Coronal (Y)", [co_lq, co_fk, co_sq])]
    titles = ["Input (LQ)", "Generated (HQ)", "Truth (HQ)"]
    
    for r, (row_name, imgs) in enumerate(rows):
        for c, img in enumerate(imgs):
            ax = axes[r, c]
            ax.imshow(img, cmap='gray', vmin=-60, vmax=0, aspect='auto')
            if r==0: ax.set_title(titles[c], fontsize=18, fontweight='bold')
            if c==0: ax.set_ylabel(row_name, fontsize=18, fontweight='bold')
            ax.axis('off')
            
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close(fig)

if __name__ == '__main__':
    opt = TestOptions().parse()
    opt.num_threads = 0
    opt.batch_size = 1
    opt.serial_batches = True
    opt.no_flip = True
    opt.display_id = -1
    
    # 在 2.5D 中，网络输入必须是 3 通道
    opt.input_nc = 3

    print("\n" + "="*80)
    print(f"🚀 全卷测试 (2.5D Slice-by-Slice): {opt.name}")
    print("="*80)

    model = create_model(opt)
    model.setup(opt)
    
    save_dir = os.path.join(opt.results_dir, opt.name)
    if not os.path.exists(save_dir): os.makedirs(save_dir)
    
    dir_lq = os.path.join(opt.dataroot, opt.dir_lq)
    dir_sq = os.path.join(opt.dataroot, opt.dir_sq)
    
    files_lq = sorted([f for f in os.listdir(dir_lq) if f.endswith('LQ.nii') or f.endswith('LQ.nii.gz')])
    
    print(f"📊 找到 {len(files_lq)} 个 LQ 测试文件")
    
    for i, fname_lq in enumerate(files_lq):
        if i >= opt.num_test: break
        
        fname_sq = fname_lq.replace('_LQ', '_HQ')
        path_lq = os.path.join(dir_lq, fname_lq)
        path_sq = os.path.join(dir_sq, fname_sq)
        
        has_truth = os.path.exists(path_sq)
        case_name = fname_lq.replace('_LQ.nii.gz', '').replace('_LQ.nii', '')
        
        print(f"\nProcessing [{i+1}]: {case_name}")
        
        # 1. 读取原始数据和珍贵的 0.036mm Affine
        nii_lq = nib.load(path_lq)
        affine = nii_lq.affine 
        vol_lq = nii_lq.get_fdata().astype(np.float32)
        orig_shape = vol_lq.shape
        
        # 2. 维度检查与转置 
        transposed = False
        if vol_lq.shape[2] > vol_lq.shape[0] and vol_lq.shape[2] > vol_lq.shape[1]:
            print("  -> Transposing to (D, H, W) for slice-by-slice inference...")
            vol_lq = vol_lq.transpose(2, 1, 0)
            transposed = True
            
        if has_truth:
            nii_sq = nib.load(path_sq)
            vol_sq = nii_sq.get_fdata().astype(np.float32)
            if transposed: vol_sq = vol_sq.transpose(2, 1, 0)
        else:
            vol_sq = None
            print(f"⚠️  未找到真值文件: {fname_sq}")

        # ==========================================================
        # 3. [核心执行] 极速 2.5D 切片扫描
        # ==========================================================
        print(f"  -> Scanning {vol_lq.shape[0]} slices...")
        vol_fake = predict_slice_by_slice(model, vol_lq, opt)
        
        # 4. 生成对比图
        if has_truth:
            psnr = calculate_psnr(vol_fake, vol_sq)
            print(f"  ✅ PSNR: {psnr:.2f} dB")
            save_paper_fig_9grid(
                os.path.join(save_dir, f"{case_name}_{opt.name}_Comparison.png"),
                case_name, opt.name, {'PSNR': psnr},
                vol_lq, vol_fake, vol_sq
            )
        
        # 5. 还原并保存 (Resave All 策略)
        if transposed:
            print("  -> Restoring shape for saving...")
            vol_fake_save = vol_fake.transpose(2, 1, 0)
            vol_lq_save   = vol_lq.transpose(2, 1, 0)
            if has_truth: vol_sq_save = vol_sq.transpose(2, 1, 0)
        else:
            vol_fake_save = vol_fake
            vol_lq_save   = vol_lq
            vol_sq_save   = vol_sq
            
        print(f"  Saving shape: {vol_fake_save.shape} (Matches Orig: {orig_shape})")
        
        save_nii(vol_fake_save, os.path.join(save_dir, f"{case_name}_{opt.name}_Fake.nii"), affine)
        save_nii(vol_lq_save,   os.path.join(save_dir, f"{case_name}_{opt.name}_Input.nii"), affine)
        if has_truth:
            save_nii(vol_sq_save, os.path.join(save_dir, f"{case_name}_{opt.name}_Truth.nii"), affine)
            
    print(f"\n✅ 完成! 所有极清无拉扯的 3D 数据已保存至: {save_dir}")