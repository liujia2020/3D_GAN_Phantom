import nibabel as nib
import numpy as np
import os
from tqdm import tqdm

# ================= 配置区域 (WSL路径) =================
# 输入文件夹 (Windows的 G:\Simu_data_1channel)
SOURCE_ROOT = "/mnt/g/Simu_data_1channel"

# 输出文件夹 (Windows的 G:\Simu_data_1channel_45db)
# TARGET_ROOT = "/mnt/g/Simu_data_1channel_45db"
TARGET_ROOT = "/mnt/g/Simu_data_1channel_30db"

# 处理子文件夹
SUB_DIRS = ["Recon_LQ_03", "Recon_SQ_75"]

# 阈值设置
CLIP_MIN = -30.0  # 低于 -45 的全变成 -45
CLIP_MAX = 0.0    # 高于 0 的全变成 0
# ====================================================

def process_nifti_clip_only(src_path, dst_path):
    try:
        # 1. 读取
        nii = nib.load(src_path)
        data = nii.get_fdata()

        # 2. 纯截断 (不归一化，不改变原本的大小关系)
        # 结果范围依然是 [-45.0, 0.0]
        data_clipped = np.clip(data, CLIP_MIN, CLIP_MAX)

        # 3. 保存 (保持浮点精度)
        new_nii = nib.Nifti1Image(data_clipped.astype(np.float32), nii.affine, nii.header)
        nib.save(new_nii, dst_path)
        
    except Exception as e:
        print(f"❌ 错误: {src_path} - {e}")

def main():
    print(f"🚀 开始处理: 只截断到 [{CLIP_MIN}, {CLIP_MAX}]，不归一化")
    
    if not os.path.exists(TARGET_ROOT):
        os.makedirs(TARGET_ROOT)

    for sub_dir in SUB_DIRS:
        src_dir = os.path.join(SOURCE_ROOT, sub_dir)
        dst_dir = os.path.join(TARGET_ROOT, sub_dir)

        if not os.path.exists(dst_dir):
            os.makedirs(dst_dir)

        if os.path.exists(src_dir):
            files = [f for f in os.listdir(src_dir) if f.endswith('.nii') or f.endswith('.nii.gz')]
            print(f"\n📂 处理文件夹: {sub_dir} (共 {len(files)} 个文件)")
            
            for f in tqdm(files):
                process_nifti_clip_only(
                    os.path.join(src_dir, f), 
                    os.path.join(dst_dir, f)
                )
        else:
            print(f"⚠️ 找不到源文件夹: {src_dir}")

    print("\n✅ 完成。数据范围现在是 -45.0 到 0.0。")

if __name__ == "__main__":
    main()