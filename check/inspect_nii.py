import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import os

# ==========================================
# 请在这里填入你随便一个真实文件的路径
# 建议分别测一个 LQ 和一个 SQ
# ==========================================
file_path = "/home/liujia/g_linux/Simu_1channel/Recon_LQ_03/SimData_NII_0001_Pts_282_lq_3ang_dB.nii"  # <--- 修改这里

def inspect_nii(path):
    if not os.path.exists(path):
        print(f"❌ 文件不存在: {path}")
        return

    try:
        # 读取 NII
        img = nib.load(path)
        data = img.get_fdata()
        
        # 统计信息
        print(f"📂 文件名: {os.path.basename(path)}")
        print(f"📏 尺寸: {data.shape}")
        print(f"📊 数值范围: Min = {data.min():.4f}, Max = {data.max():.4f}")
        print(f"Av 平均值: {data.mean():.4f}")
        
        # 物理意义推断
        if data.max() > 10.0: 
            print("💡 推断: 可能是原始线性数据 (未Log)，数值很大。")
        elif data.max() <= 0.0 and data.min() >= -100:
            print("💡 推断: 看起来像是标准的 dB 数据 (0 是最亮, 负数是变暗)。")
            if np.isclose(data.max(), 0.0, atol=1e-1):
                 print("   ✅ 确认: 0dB 是最大值（白色）。")
            else:
                 print(f"   ⚠️ 注意: 最大值不是 0，而是 {data.max()}，可能没归一化到 0dB。")
        else:
            print("💡 推断: 数值范围比较奇怪，请检查。")

        # 画个直方图看看分布
        plt.figure(figsize=(10, 4))
        plt.hist(data.flatten(), bins=100, color='blue', alpha=0.7)
        plt.title(f"Histogram of {os.path.basename(path)}")
        plt.xlabel("Pixel Value")
        plt.ylabel("Frequency")
        plt.yscale('log') # 用对数坐标看，因为背景点太多
        plt.grid(True)
        plt.show()
        
    except Exception as e:
        print(f"❌ 读取出错: {e}")

# 运行
inspect_nii(file_path)