"""
test.py - 最终完美版 (Resave All Strategy)
功能：
1. 强制重存 Input 和 Truth，确保与 Fake 坐标信息 100% 一致 (解决 3D Slicer 对齐问题)。
2. 自动匹配 _LQ / _HQ 文件名。
3. 智能处理维度转置与还原。
4. 生成带实验名后缀的文件和对比图。
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
from util import save_nii, predict_sliding_window

def calculate_psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0: return 100
    return 20 * np.log10(255.0 / np.sqrt(mse))

def save_paper_fig_9grid(save_path, case_name, model_name, metrics_dict, vol_lq, vol_fake, vol_sq):
    # 这里的输入必须是 (D, H, W) 格式，否则切面会切错
    D, H, W = vol_lq.shape
    idx_z = 500 if D > 500 else D // 2
    idx_x = 64  if W > 64  else W // 2
    idx_y = 64  if H > 64  else H // 2
    
    # 提取切片
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
    
    patch_size = (opt.patch_size_d, opt.patch_size_h, opt.patch_size_w)
    stride = (patch_size[0]//2, patch_size[1]//2, patch_size[2]//2)

    print("\n" + "="*80)
    print(f"🚀 全卷测试 (Resave Mode): {opt.name}")
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
        
        # 1. 自动匹配文件名
        fname_sq = fname_lq.replace('_LQ', '_HQ')
        path_lq = os.path.join(dir_lq, fname_lq)
        path_sq = os.path.join(dir_sq, fname_sq)
        
        has_truth = os.path.exists(path_sq)
        case_name = fname_lq.replace('_LQ.nii.gz', '').replace('_LQ.nii', '')
        
        print(f"\nProcessing [{i+1}]: {case_name}")
        
        # 2. 读取 Input
        nii_lq = nib.load(path_lq)
        affine = nii_lq.affine # <--- 锁定这个 affine，所有人保存都用它！
        vol_lq = nii_lq.get_fdata().astype(np.float32)
        orig_shape = vol_lq.shape
        
        # 3. 维度检查与转置 (为网络推理准备)
        transposed = False
        # 如果 Z 轴在最后 (H, W, D)，转置为 (D, H, W)
        if vol_lq.shape[2] > vol_lq.shape[0] and vol_lq.shape[2] > vol_lq.shape[1]:
            print("  -> Transposing to (D, H, W) for inference...")
            vol_lq = vol_lq.transpose(2, 1, 0)
            transposed = True
            
        # 4. 读取 Truth (如果有)
        if has_truth:
            nii_sq = nib.load(path_sq)
            vol_sq = nii_sq.get_fdata().astype(np.float32)
            if transposed: vol_sq = vol_sq.transpose(2, 1, 0)
        else:
            vol_sq = None
            print(f"⚠️  未找到真值文件: {fname_sq}")

        # 5. 滑窗推理 (得到 vol_fake)
        # 此时 vol_fake, vol_lq, vol_sq 都是 (D, H, W) 格式，适合画切片图
        vol_fake = predict_sliding_window(model, vol_lq, patch_size, stride)
        
        # 6. 生成对比图 (用 D, H, W 数据画图)
        if has_truth:
            psnr = calculate_psnr(vol_fake, vol_sq)
            print(f"  ✅ PSNR: {psnr:.2f} dB")
            save_paper_fig_9grid(
                os.path.join(save_dir, f"{case_name}_{opt.name}_Comparison.png"),
                case_name, opt.name, {'PSNR': psnr},
                vol_lq, vol_fake, vol_sq
            )
        
        # 7. 准备保存的数据 (关键：还原形状 + Resave All)
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
        
        # 8. 保存所有文件 (使用同一个 affine，确保绝对对齐)
        # 保存 Fake
        save_nii(vol_fake_save, os.path.join(save_dir, f"{case_name}_{opt.name}_Fake.nii"), affine)
        # 重存 Input
        save_nii(vol_lq_save,   os.path.join(save_dir, f"{case_name}_{opt.name}_Input.nii"), affine)
        # 重存 Truth (如果有)
        if has_truth:
            save_nii(vol_sq_save, os.path.join(save_dir, f"{case_name}_{opt.name}_Truth.nii"), affine)
            
    print(f"\n✅ 完成! 所有对齐文件已保存至: {save_dir}")