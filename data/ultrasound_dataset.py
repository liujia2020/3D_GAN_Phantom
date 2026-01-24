import os
import random
import numpy as np
import torch
import nibabel as nib
from data.base_dataset import BaseDataset

class UltrasoundDataset(BaseDataset):
    """
    读取配对的超声数据 (LQ 和 SQ)
    Folder Structure:
        dataroot/
            Recon_LQ_03/  (Input)
            Recon_SQ_75/  (Ground Truth)
    Matching:
        Based on 'SimData_NII_xxxx_Pts_xxx' pattern.
    Normalization:
        Input is dB data [-60, 0].
        Map to [-1, 1] for network input.
    """

    def __init__(self, opt):
        BaseDataset.__init__(self, opt)
        
        # 1. 确定文件夹路径
        self.dir_lq = os.path.join(opt.dataroot, opt.dir_lq)
        self.dir_sq = os.path.join(opt.dataroot, opt.dir_sq)
        
        # 2. 扫描文件并配对
        self.files = self._make_dataset_paired(self.dir_lq, self.dir_sq)
        self.size = len(self.files)
        
        # 3. 裁剪尺寸
        self.crop_d = opt.patch_size_d
        self.crop_h = opt.patch_size_h
        self.crop_w = opt.patch_size_w
        
        # 4. 归一化参数 (从 options 读取，默认 -60 到 0)
        self.min_db = opt.norm_min # -60.0
        self.max_db = opt.norm_max # 0.0

    def _make_dataset_paired(self, dir_lq, dir_sq):
        """
        Scanning and pairing logic.
        LQ: SimData_NII_0001_Pts_282_lq_3ang_dB.nii
        SQ: SimData_NII_0001_Pts_282_sq_75ang_dB.nii
        ID: SimData_NII_0001_Pts_282
        """
        pairs = []
        if not os.path.isdir(dir_lq) or not os.path.isdir(dir_sq):
            raise RuntimeError(f"Data directories not found: {dir_lq} or {dir_sq}")

        lq_files = sorted([f for f in os.listdir(dir_lq) if f.endswith('.nii')])
        sq_files = sorted([f for f in os.listdir(dir_sq) if f.endswith('.nii')])
        
        # Build a lookup dictionary for SQ files: ID -> Filename
        sq_lookup = {}
        for f in sq_files:
            # Split by '_sq_' to get ID
            if '_sq_' in f:
                case_id = f.split('_sq_')[0]
                sq_lookup[case_id] = f
        
        # Pair up
        for lq_f in lq_files:
            if '_lq_' in lq_f:
                case_id = lq_f.split('_lq_')[0]
                if case_id in sq_lookup:
                    pairs.append({
                        'lq_path': os.path.join(dir_lq, lq_f),
                        'sq_path': os.path.join(dir_sq, sq_lookup[case_id]),
                        'case_name': case_id
                    })
        
        print(f"Dataset V2 (Paired). Found {len(pairs)} pairs in {dir_lq} and {dir_sq}.")
        return pairs

    def _read_and_process(self, path):
        """
        Read NII -> Clip -> Normalize to [-1, 1]
        """
        try:
            # 读取
            img = nib.load(path)
            data = img.get_fdata().astype(np.float32) # [W, H, D] or [D, H, W] depending on save
            
            # 确保维度顺序 [D, H, W]
            # nibabel 读出来通常是 [x, y, z]，我们需要确认一下
            # 如果你的数据是 1024x128x128 (D, H, W)，nibabel 可能读成 (128, 128, 1024)
            if data.shape[2] == 1024: 
                data = data.transpose(2, 1, 0) # -> [1024, 128, 128]
            elif data.shape[0] == 1024:
                pass # 已经是 [D, H, W]
            
            # 随机裁剪 (Training only, or handle validation separately)
            # 这里简单起见，做随机裁剪。
            d, h, w = data.shape
            sd = random.randint(0, max(0, d - self.crop_d))
            sh = random.randint(0, max(0, h - self.crop_h))
            sw = random.randint(0, max(0, w - self.crop_w))
            
            crop = data[sd:sd+self.crop_d, sh:sh+self.crop_h, sw:sw+self.crop_w]
            
            # Padding if too small
            if crop.shape != (self.crop_d, self.crop_h, self.crop_w):
                pad_d = self.crop_d - crop.shape[0]
                pad_h = self.crop_h - crop.shape[1]
                pad_w = self.crop_w - crop.shape[2]
                crop = np.pad(crop, ((0,pad_d),(0,pad_h),(0,pad_w)), 'constant', constant_values=self.min_db)

            # === 归一化逻辑 ===
            # Range: [-60, 0] -> [-1, 1]
            
            # 1. Clip (钳位)
            crop = np.clip(crop, self.min_db, self.max_db)
            
            # 2. Normalize to [0, 1]
            # (x - min) / (max - min) => (x - (-60)) / 60 => (x + 60) / 60
            range_span = self.max_db - self.min_db
            norm = (crop - self.min_db) / range_span
            
            # 3. Scale to [-1, 1]
            norm = (norm - 0.5) * 2.0
            
            return norm

        except Exception as e:
            print(f"Error reading {path}: {e}")
            return np.zeros((self.crop_d, self.crop_h, self.crop_w), dtype=np.float32) - 1.0

    def __getitem__(self, idx):
        file_info = self.files[idx]
        
        # 读取数据
        lq_data = self._read_and_process(file_info['lq_path'])
        sq_data = self._read_and_process(file_info['sq_path'])
        
        # 转 Tensor [Channel, D, H, W]
        lq_tensor = torch.from_numpy(lq_data).unsqueeze(0).float()
        sq_tensor = torch.from_numpy(sq_data).unsqueeze(0).float()
        
        return {
            'lq': lq_tensor,
            'sq': sq_tensor, # Ground Truth
            'hq': sq_tensor, # 兼容代码里可能用到的 'hq' 键
            'lq_path': file_info['lq_path'],
            'sq_path': file_info['sq_path'],
            'case_name': file_info['case_name']
        }

    def __len__(self):
        return self.size

# import os
# import re
# import random
# import numpy as np
# import torch
# import torch.utils.data as data  # [修改] 直接引入 PyTorch 原生 Dataset

# class UltrasoundDataset(data.Dataset): # [修改] 不再继承 BaseDataset
#     """
#     [V29.0 - 物理定标最终版 (独立合并版)]
#     无需 base_dataset.py，直接继承 PyTorch Dataset。
    
#     功能：
#     1. Input Max ~ 4e5  -> 归一化基准 5e5
#     2. Target Max ~ 2.3e7 -> 归一化基准 2.5e7
#     3. 信号全为正 (B-mode) -> 线性映射 [0, Max] -> [-1, 1]
#     """

#     def __init__(self, opt):
#         super(UltrasoundDataset, self).__init__() # [修改] 初始化原生 Dataset
#         self.opt = opt
#         self.root_dir = opt.dataroot
        
#         # [新增] 兼容性接口：某些代码可能会调用 name()
#         self.name = 'UltrasoundDataset'
        
#         # 扫描文件列表
#         self.file_list = self._scan_files()
        
#         # Patch 设置 (从 opt 获取，如果没有就默认 128/64/64)
#         self.crop_d = getattr(opt, 'patch_size_d', 128)
#         self.crop_h = getattr(opt, 'patch_size_h', 64)
#         self.crop_w = getattr(opt, 'patch_size_w', 64)
        
#         # 物理归一化常数 (根据诊断报告设定)
#         self.NORM_MAX_INPUT = 500000.0   # 输入数据的归一化分母 (5e5)
#         self.NORM_MAX_TARGET = 25000000.0 # 真值数据的归一化分母 (2.5e7)
        
#         self.size = len(self.file_list)
#         if self.size == 0:
#             raise RuntimeError(f"未找到完整数据: {self.root_dir}")
#         print(f"[{'TRAIN' if opt.isTrain else 'VAL'}] Dataset V29.0 (Standalone). Size: {self.size}")

#     def _scan_files(self):
#         files_map = {}
#         # 宽容匹配 .nii 文件
#         pattern = re.compile(r'([a-zA-Z0-9]+)_([a-zA-Z0-9]+)_(In_n15|In_000|In_p15|Ref_HQ|GT_SQ)\.nii')
#         suffix_map = {'In_n15':'p_in_n15', 'In_000':'p_in_000', 'In_p15':'p_in_p15', 'Ref_HQ':'p_hq', 'GT_SQ':'p_sq'}

#         if not os.path.exists(self.root_dir): return []

#         for f in os.listdir(self.root_dir):
#             if not f.endswith('.nii'): continue
#             match = pattern.match(f)
#             if match:
#                 prefix_type, cid, suffix = match.groups()
#                 key = f"{prefix_type}_{cid}"
#                 if key not in files_map: files_map[key] = {'case_name': key}
#                 if suffix in suffix_map: files_map[key][suffix_map[suffix]] = os.path.join(self.root_dir, f)
        
#         valid_list = []
#         required = ['p_in_n15', 'p_in_000', 'p_in_p15', 'p_hq', 'p_sq']
#         for key, paths in files_map.items():
#             if all(k in paths for k in required): valid_list.append(paths)
#         valid_list.sort(key=lambda x: x['case_name'])
#         return valid_list

#     def __len__(self):
#         return self.size

#     def _read_and_process(self, path, slice_objs, max_val):
#         """
#         核心读取函数：带物理定标的归一化
#         """
#         try:
#             with open(path, 'rb') as f:
#                 f.seek(352) # 跳过 NIfTI header (352 bytes)
#                 raw = f.read()
#             data = np.frombuffer(raw, dtype=np.float32)
            
#             # 这里的 reshape 对应你之前确认的 1024x128x128 (F-order)
#             if data.size == 16777216:
#                 data = data.reshape((1024, 128, 128), order='F')
#             else:
#                 # 异常尺寸处理
#                 return np.zeros((self.crop_d, self.crop_h, self.crop_w), dtype=np.float32)

#             crop = data[slice_objs]

#             # 1. 物理归一化
#             norm_data = crop / max_val
            
#             # 2. 钳位 (Clip)
#             norm_data = np.clip(norm_data, 0.0, 1.0)

#             # 3. 映射到 GAN 的 [-1, 1] 区间
#             norm_data = (norm_data - 0.5) * 2.0
            
#             return norm_data

#         except Exception as e:
#             # 读取失败返回全背景 (-1.0)
#             return np.ones((self.crop_d, self.crop_h, self.crop_w), dtype=np.float32) * -1.0

#     def __getitem__(self, idx):
#         # 尝试读取，如果当前 idx 有问题则随机换一个
#         for _ in range(3):
#             try:
#                 files = self.file_list[idx]
                
#                 src_d, src_h, src_w = 1024, 128, 128
#                 # 随机裁剪坐标生成
#                 start_d = random.randint(0, max(0, src_d - self.crop_d))
#                 start_h = random.randint(0, max(0, src_h - self.crop_h))
#                 start_w = random.randint(0, max(0, src_w - self.crop_w))
                
#                 s_obj = (
#                     slice(start_d, start_d + self.crop_d),
#                     slice(start_h, start_h + self.crop_h),
#                     slice(start_w, start_w + self.crop_w)
#                 )
                
#                 # === 读取并归一化 ===
#                 # Inputs 使用 5e5
#                 n15 = self._read_and_process(files['p_in_n15'], s_obj, self.NORM_MAX_INPUT)
#                 z00 = self._read_and_process(files['p_in_000'], s_obj, self.NORM_MAX_INPUT)
#                 p15 = self._read_and_process(files['p_in_p15'], s_obj, self.NORM_MAX_INPUT)
                
#                 # Targets 使用 2.5e7
#                 hq  = self._read_and_process(files['p_hq'],     s_obj, self.NORM_MAX_TARGET)
#                 sq  = self._read_and_process(files['p_sq'],     s_obj, self.NORM_MAX_TARGET)

#                 # 堆叠输入: [3, D, H, W]
#                 input_stack = np.stack([n15, z00, p15], axis=0)
                
#                 return {
#                     'lq': torch.from_numpy(input_stack).float(),       
#                     'hq': torch.from_numpy(hq).unsqueeze(0).float(), # [1, D, H, W]
#                     'sq': torch.from_numpy(sq).unsqueeze(0).float(), # [1, D, H, W]
#                     'case_name': files['case_name']
#                 }
#             except Exception:
#                 idx = random.randint(0, self.size - 1)
        
#         # 如果三次都失败，抛出错误
#         raise RuntimeError("Load Error")

#     # [新增] 可选：modify_commandline_options 接口
#     # 如果 train_options.py 里有调用它，加上这个空函数就不会报错
#     @staticmethod
#     def modify_commandline_options(parser, is_train):
#         return parser