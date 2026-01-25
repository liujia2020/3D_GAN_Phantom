import nibabel as nib
import numpy as np
import os
import sys

# 请修改为您实际的文件路径
path_lq   = "/home/liujia/g_linux/test/simu_1channel/test_lq/Simu_001_LQ.nii"          # 原始输入
path_sq   = "/home/liujia/g_linux/test/simu_1channel/test_sq/Simu_001_HQ.nii"          # 原始真值
# 请修改为您刚刚生成的某一个结果文件路径
path_fake = "/mnt/g/train_data/results/06_1Ch_L1_Only/Simu_001_06_1Ch_L1_Only_Fake.nii"

paths = {'LQ (Original)': path_lq, 'HQ (Original)': path_sq, 'Fake (Generated)': path_fake}

print(f"{'File':<20} | {'Shape':<20} | {'Affine (Diagonal)':<30}")
print("-" * 80)

for name, p in paths.items():
    if not os.path.exists(p):
        print(f"{name:<20} | File not found!")
        continue
        
    img = nib.load(p)
    shape = img.shape
    affine = img.affine
    
    # 我们主要看对角线元素 (代表 spacing 和方向) 和最后一列 (代表原点 origin)
    diag = np.diag(affine)[:3]
    origin = affine[:3, 3]
    
    print(f"{name:<20} | {str(shape):<20} | Spacing: {diag}")
    print(f"{'':<20} | {'':<20} | Origin : {origin}")
    print("-" * 80)
    
    # 如果想看完整的 affine，取消下面注释
    # print(f"--- Full Affine for {name} ---")
    # print(affine)
    # print("\n")