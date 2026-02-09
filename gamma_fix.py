import nibabel as nib
import numpy as np
import sys
import os

def apply_gamma_correction(nii_path, save_path, gamma=2.0):
    print(f"正在处理: {nii_path}")
    print(f"Gamma 值: {gamma}")
    
    try:
        img = nib.load(nii_path)
        data = img.get_fdata()
        
        # 1. 归一化到 [0, 1]
        # (因为 Gamma 校正必须在 0-1 之间做才能压制暗部)
        data_min = data.min()
        data_max = data.max()
        print(f"原始数据范围: [{data_min:.4f}, {data_max:.4f}]")
        
        norm_data = (data - data_min) / (data_max - data_min + 1e-8)
        
        # 2. 执行 Gamma 校正 (核心步骤)
        # 伪影(0.1) -> 0.1^2 = 0.01 (消失)
        # 肌肉(0.5) -> 0.5^2 = 0.25 (保留)
        corrected_data = np.power(norm_data, gamma)
        
        # 3. 还原强度 (可选，这里我们直接保存归一化后的结果，通常更适合显示)
        # final_data = corrected_data * (data_max - data_min) + data_min
        final_data = corrected_data

        # 保存
        new_img = nib.Nifti1Image(final_data, img.affine, img.header)
        nib.save(new_img, save_path)
        print(f"✅ 处理完成! 输出文件: {save_path}")
        
    except Exception as e:
        print(f"❌ 出错了: {e}")

if __name__ == '__main__':
    # 命令行参数: python gamma_fix.py <输入> <输出> <Gamma值>
    if len(sys.argv) < 3:
        print("用法: python gamma_fix.py <input.nii> <output.nii> [gamma_value]")
    else:
        in_path = sys.argv[1]
        out_path = sys.argv[2]
        # 如果不输第3个参数，默认用 2.0
        g = float(sys.argv[3]) if len(sys.argv) > 3 else 2.0
        apply_gamma_correction(in_path, out_path, g)