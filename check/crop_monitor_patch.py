import nibabel as nib
import numpy as np
import os

def crop_patch(input_path, output_path, center_z, center_y, center_x, patch_size):
    # 1. 读取原始大体积 NIfTI 文件
    print(f"正在读取: {input_path}")
    img = nib.load(input_path)
    data = img.get_fdata().astype(np.float32)
    affine = img.affine
    
    # 2. 计算裁剪边界 (基于 D, H, W = 128, 64, 64)
    d, h, w = patch_size
    z_start = center_z - d // 2
    z_end = center_z + d // 2
    y_start = center_y - h // 2
    y_end = center_y + h // 2
    x_start = center_x - w // 2
    x_end = center_x + w // 2
    
    # 3. 矩阵切片裁剪
    cropped_data = data[z_start:z_end, y_start:y_end, x_start:x_end]
    
    # 4. 保存为定制的监控文件 (取消 .gz 压缩，直接存 .nii)
    cropped_img = nib.Nifti1Image(cropped_data, affine)
    nib.save(cropped_img, output_path)
    print(f"✅ 成功保存: {output_path} (Shape: {cropped_data.shape})")

if __name__ == "__main__":
    # 自动转换的 WSL 本地路径
    LQ_INPUT = "/home/liujia/g_linux/test/simu_1channel/Recon_LQ_03/Phantom_100_LQ.nii"  
    SQ_INPUT = "/home/liujia/g_linux/test/simu_1channel/Recon_SQ_75/Phantom_100_HQ.nii"  
    
    # 将监控文件保存在项目下的一个独立文件夹里
    OUTPUT_DIR = "./monitor_data"
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    # 保存名称直接用 .nii
    LQ_OUTPUT = os.path.join(OUTPUT_DIR, "monitor_LQ.nii")
    SQ_OUTPUT = os.path.join(OUTPUT_DIR, "monitor_HQ.nii")
    
    # 确定的完美坐标 (D, H, W)
    CENTER_Z = 720
    CENTER_Y = 89
    CENTER_X = 69
    PATCH_SIZE = (128, 64, 64) 
    
    print("开始精准裁剪监控数据...")
    
    if os.path.exists(LQ_INPUT):
        crop_patch(LQ_INPUT, LQ_OUTPUT, CENTER_Z, CENTER_Y, CENTER_X, PATCH_SIZE)
    else:
        print(f"⚠️ 找不到 LQ 文件，请检查路径: {LQ_INPUT}")
        
    if os.path.exists(SQ_INPUT):
        crop_patch(SQ_INPUT, SQ_OUTPUT, CENTER_Z, CENTER_Y, CENTER_X, PATCH_SIZE)
    else:
        print(f"⚠️ 找不到 HQ 文件，请检查路径: {SQ_INPUT}")