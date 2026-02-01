# =========================================================================
# Compare_By_ID_WSL.py
# 功能：通过提取文件名中的 ID (如 0001) 来强制配对新旧文件
#       忽略文件名中 Pts、Angle 等参数的差异，专注于对比 ID 相同的文件
# 环境：WSL (Ubuntu)
# =========================================================================

import os
import glob
import re
import numpy as np
import nibabel as nib

# ================= 配置区域 (WSL 路径) =================
# 旧文件夹 (基准)
DIR_OLD = '/home/liujia/g_linux/Simu_1channel'

# 新文件夹 (待验证)
DIR_NEW = '/home/liujia/g_linux/Simu2'

# 需要对比的子文件夹
SUB_FOLDERS = ['Recon_LQ_03', 'Recon_SQ_75']
# ======================================================

def extract_id(filename):
    """
    从文件名中提取 ID
    例如: Simu_Data_NII_0001_Pts_... -> 返回 '0001'
    例如: SimData_NII_0001_Pts_...   -> 返回 '0001'
    """
    # 匹配 NII_ 后面的数字，或者直接匹配 4位数字
    match = re.search(r'NII_(\d+)_', filename)
    if match:
        return match.group(1)
    
    # 备用方案：如果名字里没有 NII_，尝试找连续的数字
    match_fallback = re.search(r'(\d+)', filename)
    if match_fallback:
        return match_fallback.group(1)
    
    return None

def compare_header_only(file_old, file_new):
    """只对比头文件信息，不对比数据内容(因为内容肯定不一样)"""
    try:
        nii_old = nib.load(file_old)
        nii_new = nib.load(file_new)
    except Exception as e:
        print(f"❌ [无法读取] {e}")
        return

    h_old = nii_old.header
    h_new = nii_new.header

    # 1. 维度对比
    shape_old = h_old.get_data_shape()
    shape_new = h_new.get_data_shape()

    # 2. 间距对比
    zoom_old = h_old.get_zooms()
    zoom_new = h_new.get_zooms()

    print(f"   [Old] Dim: {shape_old} | Spacing: {zoom_old}")
    print(f"   [New] Dim: {shape_new} | Spacing: {zoom_new}")

    # 判断维度是否只是转置了
    if shape_old != shape_new:
        if sorted(shape_old) == sorted(shape_new):
            print(f"   ⚠️  [注意] 维度发生了转置 (正常现象)")
        else:
            print(f"   ❌ [警告] 维度完全不匹配!")
    
    # 判断间距
    if not np.allclose(zoom_old, zoom_new, atol=1e-4):
        # 如果维度转置了，间距也应该转置，这里检查一下是否对应
        if sorted(zoom_old) == sorted(zoom_new):
            print(f"   ⚠️  [注意] 间距也跟随维度发生了转置 (正常现象)")
        else:
            print(f"   ❌ [警告] 间距数值不匹配!")

def process_folder(sub_folder):
    path_new_root = os.path.join(DIR_NEW, sub_folder)
    path_old_root = os.path.join(DIR_OLD, sub_folder)
    
    print(f"\n{'='*20} 正在检查: {sub_folder} {'='*20}")
    
    # 1. 获取所有旧文件，并建立 {ID: 路径} 的索引
    old_files_map = {}
    if os.path.exists(path_old_root):
        # 递归查找所有 nii
        for f in glob.glob(os.path.join(path_old_root, '**', '*.nii'), recursive=True):
            fname = os.path.basename(f)
            fid = extract_id(fname)
            if fid:
                old_files_map[fid] = f
    else:
        print(f"❌ 旧文件夹不存在: {path_old_root}")
        return

    if not old_files_map:
        print("⚠️ 旧文件夹里没找到 NIfTI 文件")
        return

    # 2. 遍历新文件去匹配
    new_files = glob.glob(os.path.join(path_new_root, '*.nii'))
    if not new_files:
        print(f"⚠️ 新文件夹里没找到 NIfTI 文件: {path_new_root}")
        return

    count = 0
    for f_new in new_files:
        fname_new = os.path.basename(f_new)
        fid_new = extract_id(fname_new)
        
        if not fid_new:
            print(f"⚠️ 跳过无法识别 ID 的文件: {fname_new}")
            continue
            
        # 在旧映射里找
        if fid_new in old_files_map:
            f_old = old_files_map[fid_new]
            fname_old = os.path.basename(f_old)
            
            print(f"🆔 对比 ID [{fid_new}]")
            print(f"   Old: {fname_old}")
            print(f"   New: {fname_new}")
            
            compare_header_only(f_old, f_new)
            count += 1
        else:
            print(f"⚠️ ID [{fid_new}] 在旧库中未找到对应文件 (新文件: {fname_new})")

    print(f"\n📊 共对比了 {count} 组文件")

def main():
    print("🚀 开始基于 ID 匹配对比 (WSL版)...")
    
    for sub in SUB_FOLDERS:
        process_folder(sub)

if __name__ == "__main__":
    main()