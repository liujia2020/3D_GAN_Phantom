import os
import nibabel as nib
import numpy as np
from scipy.ndimage import zoom

def convert_to_isotropic(src_dataroot, dst_dataroot, orig_z_spacing=0.0362, target_spacing=0.2):
    dirs_to_convert = ['Recon_LQ_03', 'Recon_SQ_75']
    
    # 计算 Z 轴需要缩放的比例
    scale_factor = orig_z_spacing / target_spacing
    print(f"🔄 Z轴缩放比例: {scale_factor:.4f} ({orig_z_spacing}mm -> {target_spacing}mm)")

    for dname in dirs_to_convert:
        src_dir = os.path.join(src_dataroot, dname)
        dst_dir = os.path.join(dst_dataroot, dname)
        os.makedirs(dst_dir, exist_ok=True)
        
        if not os.path.exists(src_dir):
            continue
            
        for fname in os.listdir(src_dir):
            if not (fname.endswith('.nii') or fname.endswith('.nii.gz')):
                continue
                
            src_path = os.path.join(src_dir, fname)
            dst_path = os.path.join(dst_dir, fname)
            
            # 1. 读取原始数据和仿射矩阵
            img = nib.load(src_path)
            data = img.get_fdata().astype(np.float32)
            affine = img.affine.copy()
            
            # [关键修复 1]：提取原始的坐标系标准代码，防止丢失
            sform_code = int(img.header['sform_code'])
            qform_code = int(img.header['qform_code'])
            
            # 2. 找到最长的维度 (即深度 Z 轴)
            depth_axis = np.argmax(data.shape)
            
            # 3. 设置三维缩放因子 (只压缩深度轴)
            zoom_factors = [1.0, 1.0, 1.0]
            zoom_factors[depth_axis] = scale_factor
            
            # 4. 执行插值压缩
            print(f"正在处理: {fname}")
            print(f"  -> 原尺寸: {data.shape}")
            new_data = zoom(data, zoom_factors, order=3)
            print(f"  -> 新尺寸: {new_data.shape}")
            
            # 5. 修改 Affine 矩阵
            # [关键修复 2]：对代表深度的整列进行精确缩放，确保其他轴的正负号绝对不受影响
            affine[:3, depth_axis] *= (1.0 / scale_factor)
            
            # 6. 保存为新的 NIfTI 文件
            # 将带有原始信息的 header 传入
            new_img = nib.Nifti1Image(new_data, affine, header=img.header)
            
            # [终极修复 3]：强行覆写 sform 和 qform，彻底锁死矩阵方向！
            # 这一步直接粉碎了 nibabel 乱翻转 Y 轴的企图。
            new_img.set_sform(affine, sform_code)
            new_img.set_qform(affine, qform_code)
            
            nib.save(new_img, dst_path)
            
    print("\n✅ 所有数据已成功转换为各向同性，并保存在:", dst_dataroot)

if __name__ == '__main__':
    # 请根据你的实际路径修改这里
    SOURCE_ROOT = "/home/liujia/g_linux/Phantom_Carotid_Muscle/"
    TARGET_ROOT = "/home/liujia/g_linux/Phantom_Isotropic/"
    
    convert_to_isotropic(SOURCE_ROOT, TARGET_ROOT)