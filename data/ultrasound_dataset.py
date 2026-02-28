import os
import random
import numpy as np
import torch
import nibabel as nib
from data.base_dataset import BaseDataset

class UltrasoundDataset(BaseDataset):
    """
    Dataset V8 (2.5D Slice-by-Slice Perfection):
    完全抛弃 3D 立方体切块，改为随机抽取 1 张目标切片，并附带上下两张切片作为上下文。
    输入LQ: (3, H, W)，真值SQ: (1, H, W)
    """
    def __init__(self, opt):
        BaseDataset.__init__(self, opt)
        
        self.dir_lq = os.path.join(opt.dataroot, opt.dir_lq)
        self.dir_sq = os.path.join(opt.dataroot, opt.dir_sq)
        
        # 扫描文件
        self.lq_paths = sorted([os.path.join(self.dir_lq, f) for f in os.listdir(self.dir_lq) if f.endswith('.nii') or f.endswith('.nii.gz')])
        self.sq_paths = sorted([os.path.join(self.dir_sq, f) for f in os.listdir(self.dir_sq) if f.endswith('.nii') or f.endswith('.nii.gz')])
        
        # 兼容 tif
        if len(self.lq_paths) == 0:
            self.lq_paths = sorted([os.path.join(self.dir_lq, f) for f in os.listdir(self.dir_lq) if f.endswith('.tif') or f.endswith('.tiff')])
            self.sq_paths = sorted([os.path.join(self.dir_sq, f) for f in os.listdir(self.dir_sq) if f.endswith('.tif') or f.endswith('.tiff')])

        assert len(self.lq_paths) > 0, f"未找到数据! 路径: {self.dir_lq}"
        assert len(self.lq_paths) == len(self.sq_paths), f"文件数量不匹配! LQ={len(self.lq_paths)}, SQ={len(self.sq_paths)}"
            
        print(f"Dataset V8 (2.5D Mode). Found {len(self.lq_paths)} paired volumes.")

        # 注意：patch_d 参数在这里已被废弃不用，仅使用 patch_h 和 patch_w 进行 2D 裁剪
        self.patch_h = opt.patch_size_h
        self.patch_w = opt.patch_size_w
        self.norm_min = opt.norm_min
        self.norm_max = opt.norm_max

    def _read_volume(self, path):
        if path.endswith('.nii') or path.endswith('.nii.gz'):
            img = nib.load(path)
            data = img.get_fdata().astype(np.float32)
            # 确保 Z 轴在第 0 维 (D, H, W)
            if data.shape[2] > data.shape[0] and data.shape[2] > data.shape[1]:
                data = data.transpose(2, 1, 0)
            return data
        else:
            import tifffile as tiff
            return tiff.imread(path).astype(np.float32)

    def __getitem__(self, index):
        index = index % len(self.lq_paths)
        lq_path = self.lq_paths[index]
        sq_path = self.sq_paths[index]

        # 1. 读取 3D 体数据 (D, H, W)
        img_lq = self._read_volume(lq_path)
        img_sq = self._read_volume(sq_path)
        
        d, h, w = img_lq.shape
        
        # 2. 核心 2.5D 切片抽取逻辑
        # 在 Z 轴上随机选定目标层 z
        z = random.randint(0, d - 1)
        z_prev = max(0, z - 1)       # 边界保护：第一层的上一层依然是自己
        z_next = min(d - 1, z + 1)   # 边界保护：最后一层的下一层依然是自己
        
        # 3. 2D 平面随机裁剪计算
        h_max = max(0, h - self.patch_h)
        w_max = max(0, w - self.patch_w)
        y = random.randint(0, h_max)
        x = random.randint(0, w_max)
        
        # 提取当前视窗内的 2D 切片 (H, W)
        lq_prev_slice = img_lq[z_prev, y : y + self.patch_h, x : x + self.patch_w]
        lq_curr_slice = img_lq[z,      y : y + self.patch_h, x : x + self.patch_w]
        lq_next_slice = img_lq[z_next, y : y + self.patch_h, x : x + self.patch_w]
        
        sq_curr_slice = img_sq[z,      y : y + self.patch_h, x : x + self.patch_w]
        
        # 把 LQ 的 3 张切片像夹心饼干一样叠在一起，变成 (3, H, W)
        patch_lq = np.stack([lq_prev_slice, lq_curr_slice, lq_next_slice], axis=0)
        # 真值 SQ 只需要中间那一层，扩展通道维度变成 (1, H, W)
        patch_hq = np.expand_dims(sq_curr_slice, axis=0)
        
        # Padding: 如果原图比裁剪尺寸小，需要补齐
        if patch_lq.shape[1] != self.patch_h or patch_lq.shape[2] != self.patch_w:
            patch_lq = self.pad_tensor_2d(patch_lq)
            patch_hq = self.pad_tensor_2d(patch_hq)

        # 4. 归一化
        patch_lq = np.clip(patch_lq, self.norm_min, self.norm_max)
        patch_hq = np.clip(patch_hq, self.norm_min, self.norm_max)
        
        rnge = self.norm_max - self.norm_min
        if rnge == 0: rnge = 1.0
        
        patch_lq = (patch_lq - self.norm_min) / rnge * 2.0 - 1.0
        patch_hq = (patch_hq - self.norm_min) / rnge * 2.0 - 1.0
        
        # 5. 转 Tensor
        tensor_lq = torch.from_numpy(patch_lq).float() # 已经是 (3, H, W)
        tensor_hq = torch.from_numpy(patch_hq).float() # 已经是 (1, H, W)
        
        case_name = os.path.basename(lq_path)

        return {
            'lq': tensor_lq, 
            'sq': tensor_hq, 
            'hq': tensor_hq, 
            'lq_path': lq_path, 
            'sq_path': sq_path,
            'case_name': case_name
        }

    def __len__(self):
        return len(self.lq_paths)
        
    def pad_tensor_2d(self, img):
        # 针对 2.5D 的补零函数，img shape: (C, H, W)
        c, h, w = img.shape
        pad_h = max(0, self.patch_h - h)
        pad_w = max(0, self.patch_w - w)
        return np.pad(img, ((0, 0), (0, pad_h), (0, pad_w)), 'constant', constant_values=self.norm_min)

    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser