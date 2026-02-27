import nibabel as nib
import numpy as np

# 请替换为您实际的文件路径
path_gt   = '/home/liujia/g_linux/test/simu_stand_fixed_v2/Simu_0010_GT_SQ.nii'
path_exp1 = '/mnt/g/train_data/results/06_1Ch_L1_Only/Simu_001_06_1Ch_L1_Only_Fake.nii'

def check_range(name, path):
    try:
        img = nib.load(path)
        data = img.get_fdata()
        # 排除背景(假设小于-100的都是背景)，只看有信号的区域
        valid_data = data[data > -1000] 
        print(f"[{name}]")
        print(f"  Min : {np.min(data):.4f}")
        print(f"  Max : {np.max(data):.4f}")
        print(f"  Mean: {np.mean(data):.4f}")
        
        # 检查是否有负数
        if np.min(data) < 0:
            print("  ⚠️ 警告: 数据包含负数 (可能是 dB 数据)")
        else:
            print("  ✅ 正常: 数据为非负数 (可能是线性数据)")
        print("-" * 50)
    except Exception as e:
        print(f"Error reading {name}: {e}")

check_range("Ground Truth", path_gt)
check_range("Exp1 (Fake)", path_exp1)