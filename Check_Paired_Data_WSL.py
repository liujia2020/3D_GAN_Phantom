# =========================================================================
# Check_Paired_Data_WSL.py
# 功能：检查新的训练数据集是否一一配对 (LQ vs SQ)
# 逻辑：提取 ID，确保 Recon_LQ_03 和 Recon_SQ_75 中的 ID 完全一致
# 环境：WSL
# =========================================================================

import os
import glob
import re

# ================= 配置区域 =================
# 新数据的根目录 (WSL 路径)
DATA_ROOT = '/home/liujia/g_linux/Simu2'

# 两个需要配对的文件夹
DIR_A_NAME = 'Recon_LQ_03' # 输入 (Input)
DIR_B_NAME = 'Recon_SQ_75' # 标签 (Target)
# ===========================================

def extract_id(filename):
    """
    从文件名中提取 ID
    例如: Simu_Data_NII_0001_Pts_... -> 返回 '0001'
    """
    # 匹配 NII_ 后面的数字
    match = re.search(r'NII_(\d+)_', filename)
    if match:
        return match.group(1)
    return None

def main():
    path_a = os.path.join(DATA_ROOT, DIR_A_NAME)
    path_b = os.path.join(DATA_ROOT, DIR_B_NAME)

    print(f"🚀 开始检查数据配对...")
    print(f"📂 根目录: {DATA_ROOT}")
    print(f"   文件夹 A: {DIR_A_NAME}")
    print(f"   文件夹 B: {DIR_B_NAME}\n")

    if not os.path.exists(path_a) or not os.path.exists(path_b):
        print("❌ 错误: 找不到子文件夹，请检查路径。")
        return

    # 1. 扫描 A 文件夹
    files_a = glob.glob(os.path.join(path_a, '*.nii'))
    id_map_a = {}
    for f in files_a:
        fname = os.path.basename(f)
        fid = extract_id(fname)
        if fid:
            id_map_a[fid] = fname
    
    # 2. 扫描 B 文件夹
    files_b = glob.glob(os.path.join(path_b, '*.nii'))
    id_map_b = {}
    for f in files_b:
        fname = os.path.basename(f)
        fid = extract_id(fname)
        if fid:
            id_map_b[fid] = fname

    # 3. 集合运算
    ids_a = set(id_map_a.keys())
    ids_b = set(id_map_b.keys())

    common_ids = ids_a.intersection(ids_b)
    missing_in_b = ids_a - ids_b # A 有 B 没有
    missing_in_a = ids_b - ids_a # B 有 A 没有

    # 4. 输出报告
    print("-" * 50)
    print(f"📊 统计报告:")
    print(f"   {DIR_A_NAME} 文件数: {len(ids_a)}")
    print(f"   {DIR_B_NAME} 文件数: {len(ids_b)}")
    print(f"   ✅ 完美配对数: {len(common_ids)}")
    print("-" * 50)

    # 5. 报错
    if len(missing_in_b) > 0:
        print(f"\n❌ 警告: 在 {DIR_B_NAME} 中缺失 {len(missing_in_b)} 个文件:")
        for i, fid in enumerate(sorted(list(missing_in_b))):
            print(f"   ID [{fid}] (存在于 {DIR_A_NAME}: {id_map_a[fid]})")
            if i >= 9: 
                print("   ... (更多省略)")
                break

    if len(missing_in_a) > 0:
        print(f"\n❌ 警告: 在 {DIR_A_NAME} 中缺失 {len(missing_in_a)} 个文件:")
        for i, fid in enumerate(sorted(list(missing_in_a))):
            print(f"   ID [{fid}] (存在于 {DIR_B_NAME}: {id_map_b[fid]})")
            if i >= 9:
                print("   ... (更多省略)")
                break

    if len(missing_in_a) == 0 and len(missing_in_b) == 0:
        print("\n🎉 恭喜！所有数据一一配对，无缺失。")
    else:
        print("\n⚠️  请修复上述缺失文件后再开始训练，否则 Dataloader 可能会报错。")

if __name__ == "__main__":
    main()