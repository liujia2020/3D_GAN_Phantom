import os
import re
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

# ================= 配置区域 =================
# 请修改为你的实际数据路径
DATAROOT = "/home/liujia/g_linux/test/simu_stand_fixed_v2" 
# ===========================================

def read_raw_nii(path):
    """模拟 Dataset 中的读取逻辑"""
    try:
        # 直接用 nibabel 读，或者用你代码里的二进制读取法
        # 这里为了通用简单，我们尝试用 nibabel (如果你的文件是标准 nii)
        # 如果你的文件头损坏只能用二进制读，请替换回 dataset 里的 _read_and_process 逻辑
        img = nib.load(path)
        data = img.get_fdata().astype(np.float32)
        
        # 维度修正 (根据你之前的描述)
        # 如果是 1D 数组或形状不对，强制 reshape
        if data.ndim != 3:
            # 假设数据是 1024x128x128
            data = data.reshape((1024, 128, 128), order='F')
        
        return data
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return None

def normalize(data):
    """简单的 0-1 归一化用于显示"""
    d_min = data.min()
    d_max = data.max()
    return (data - d_min) / (d_max - d_min + 1e-8)

def scan_one_case(root):
    """随便找一组完整的数据"""
    files = os.listdir(root)
    pattern = re.compile(r'([a-zA-Z0-9]+)_([a-zA-Z0-9]+)_(In_n15|In_000|In_p15)\.nii')
    
    # 找一个 case_id
    found_case = None
    for f in files:
        m = pattern.match(f)
        if m:
            found_case = f"{m.group(1)}_{m.group(2)}"
            break
    
    if not found_case:
        print("❌ 没找到任何匹配的文件！")
        return None

    print(f"🔍 正在检查 Case: {found_case}")
    paths = {
        'n15': os.path.join(root, f"{found_case}_In_n15.nii"),
        'z00': os.path.join(root, f"{found_case}_In_000.nii"),
        'p15': os.path.join(root, f"{found_case}_In_p15.nii")
    }
    return paths

def main():
    # 1. 找文件
    paths = scan_one_case(DATAROOT)
    if not paths: return

    # 2. 读取三个通道
    print("📖 读取数据中...")
    vol_n15 = read_raw_nii(paths['n15'])
    vol_z00 = read_raw_nii(paths['z00'])
    vol_p15 = read_raw_nii(paths['p15'])

    if vol_n15 is None: return

    # 3. 执行堆叠 (核心操作)
    # [3, D, H, W]
    stack_tensor = np.stack([vol_n15, vol_z00, vol_p15], axis=0)
    
    print(f"\n✅ 堆叠成功！")
    print(f"原始形状: {vol_n15.shape}")
    print(f"堆叠后形状 (Shape): {stack_tensor.shape}")
    print(f"  - Dim 0 (Channels): {stack_tensor.shape[0]} -> [n15, z00, p15]")
    print(f"  - Dim 1 (Depth):    {stack_tensor.shape[1]}")
    print(f"  - Dim 2 (Height):   {stack_tensor.shape[2]}")
    print(f"  - Dim 3 (Width):    {stack_tensor.shape[3]}")

    # 4. 可视化
    # 我们取一个中间切片来观察 (比如 Depth 方向的中间)
    # slice_idx = stack_tensor.shape[1] // 2 
    slice_idx = 501
    # 取出切片
    # [Channel, Height, Width]
    slice_n15 = normalize(stack_tensor[0, slice_idx, :, :])
    slice_z00 = normalize(stack_tensor[1, slice_idx, :, :])
    slice_p15 = normalize(stack_tensor[2, slice_idx, :, :])

    # 合成 RGB (R=n15, G=z00, B=p15)
    # 形状需要转为 [H, W, 3] 才能被 plt 显示
    rgb_img = np.stack([slice_n15, slice_z00, slice_p15], axis=-1)

    # 绘图
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 4, 1)
    plt.title("Channel 0: n15 (-15°)")
    plt.imshow(slice_n15, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 4, 2)
    plt.title("Channel 1: z00 (0°)")
    plt.imshow(slice_z00, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 4, 3)
    plt.title("Channel 2: p15 (+15°)")
    plt.imshow(slice_p15, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 4, 4)
    plt.title("RGB Composite (Difference Check)")
    plt.imshow(rgb_img)
    plt.axis('off')

    save_path = f"check_stacking_viz_{slice_idx}.png"
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    print(f"\n📸 可视化结果已保存为: {save_path}")
    print("👉 请打开这张图。")
    print("   如果 RGB 图里看到了'彩色边缘'，说明三个角度确实提供了不同的信息！")
    print("   如果 RGB 图是纯黑白的，说明三个角度数据可能重复了。")

if __name__ == '__main__':
    main()