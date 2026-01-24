import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.interpolate import make_interp_spline
import csv

# ==========================================
# 1. 配置区域 (请修改这里)
# ==========================================
# 物理间距 [Z(深度), X(宽度), Y(厚度)]
VOXEL_SPACING = np.array([0.0362, 0.2, 0.2]) 

# 初始切片位置 (深度 Z, 宽度 X, 厚度 Y)
INITIAL_POS = [500, 64, 64] 

# 文件列表 (您的路径)
file_dict = {
    'Ground Truth': '/home/liujia/g_linux/test/simu_stand_fixed_v2/Simu_0010_GT_SQ.nii', 
    'Exp1 (Pixel100)': '/mnt/g/train_data/test_results/02_augan_Pixel100_Gan0_Tv0/epoch_latest/nifti/Simu_0010_Fake.nii',
    'Exp2 (Gan5)':     '/mnt/g/train_data/test_results/03_augan_Pixel100_Gan5_Tv0/epoch_latest/nifti/Simu_0010_Fake.nii',
    'Exp3 (Tv0.1)':    '/mnt/g/train_data/test_results/04_augan_Pixel100_Gan1_Tv0.1/epoch_latest/nifti/Simu_0010_Fake.nii',
}

OUTPUT_CSV = '/mnt/g/result_ana/psf_metrics_analysis.csv'

# ==========================================
# 2. 核心逻辑
# ==========================================

def robust_load_nii(path):
    """强制二进制读取 (1024x128x128)"""
    if not os.path.exists(path):
        print(f"❌ 文件不存在: {path}")
        return None
    try:
        with open(path, 'rb') as f:
            f.seek(352)
            raw = f.read()
        data = np.frombuffer(raw, dtype=np.float32)
        
        target_size = 1024 * 128 * 128
        if data.size != target_size:
            # 简单截断或补零，保证不崩
            if data.size > target_size: data = data[:target_size]
            else:
                temp = np.zeros(target_size, dtype=np.float32)
                temp[:data.size] = data
                data = temp
        
        # 这里的 reshape 顺序必须对应 MATLAB 的存储顺序
        return data.reshape((1024, 128, 128), order='F')
    except Exception as e:
        print(f"读取错误: {e}")
        return None

def compute_metrics(profile, spacing):
    """计算 FWHM, FWTM, PSL"""
    # 插值平滑
    x = np.arange(len(profile))
    if len(profile) < 4: return 0,0,-100, x*spacing, profile
    
    x_new = np.linspace(0, len(profile)-1, len(profile)*10)
    try:
        spl = make_interp_spline(x, profile, k=3)
        y_smooth = spl(x_new)
    except:
        y_smooth = profile; x_new = x
        
    # 归一化
    y_max = y_smooth.max()
    if y_max > 1e-9: y_smooth /= y_max
    else: return 0,0,-100, x_new*spacing, y_smooth-100
    
    # 转 dB
    y_db = 20 * np.log10(np.maximum(y_smooth, 1e-5))
    
    # FWHM (-6dB)
    mask_6 = y_db >= -6.0
    if np.any(mask_6):
        idx = np.where(mask_6)[0]
        fwhm = (x_new[idx[-1]] - x_new[idx[0]]) * spacing
    else: fwhm = 0
    
    # FWTM (-20dB)
    mask_20 = y_db >= -20.0
    if np.any(mask_20):
        idx = np.where(mask_20)[0]
        fwtm = (x_new[idx[-1]] - x_new[idx[0]]) * spacing
    else: fwtm = 0
    
    # PSL (Peak Side Lobe)
    # 简单粗暴法：除去 -10dB 主瓣范围外的最大值
    mask_main = y_db >= -10.0 # 假设主瓣至少高于 -10dB
    mask_side = ~mask_main
    if np.any(mask_side):
        psl = np.max(y_db[mask_side])
    else: psl = -100
    
    return fwhm, fwtm, psl, x_new*spacing, y_db

class PSFViewer:
    def __init__(self, file_map, voxel_spacing, init_pos):
        self.file_map = file_map
        self.spacing = voxel_spacing
        self.cz, self.cx, self.cy = init_pos
        self.data_cache = {}
        self.names = []
        
        # 1. 加载数据
        print(f"🚀 初始化: 默认中心 [Z={self.cz}, X={self.cx}, Y={self.cy}]")
        for name, path in file_map.items():
            print(f"  -> 读取 {name}...")
            data = robust_load_nii(path)
            if data is not None:
                # 预处理：归一化到 0-1 以便显示，防止不同量级无法对比
                # 使用 99.9% 分位点防止噪点爆亮
                vmax = np.percentile(data, 99.9)
                if vmax > 0: data = data / vmax
                self.data_cache[name] = data
                self.names.append(name)
        
        if not self.names: raise RuntimeError("无有效数据")
        
        # 2. 创建界面 (2行3列)
        self.fig = plt.figure(figsize=(16, 10))
        gs = GridSpec(2, 3, height_ratios=[1.5, 1])
        
        # 上排：三视图
        self.ax_axial = self.fig.add_subplot(gs[0, 0]) # Z-plane (X-Y)
        self.ax_lat   = self.fig.add_subplot(gs[0, 1]) # Y-plane (Z-X)
        self.ax_ele   = self.fig.add_subplot(gs[0, 2]) # X-plane (Z-Y)
        
        # 下排：三曲线
        self.ax_prof_z = self.fig.add_subplot(gs[1, 0])
        self.ax_prof_x = self.fig.add_subplot(gs[1, 1])
        self.ax_prof_y = self.fig.add_subplot(gs[1, 2])
        
        # 连接事件
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        
        # 3. 初始绘制
        self.update_all()
        
        # CSV 头
        if not os.path.exists(OUTPUT_CSV):
            with open(OUTPUT_CSV, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Z_idx','X_idx','Y_idx','Model',
                                 'FWHM_Z','FWHM_X','FWHM_Y',
                                 'PSL_Z','PSL_X','PSL_Y'])
        
        print("\n✅ 界面就绪！请点击上排图像选择点靶。")
        plt.show()

    def update_all(self):
        """刷新所有图像和曲线"""
        self.plot_slices()
        self.plot_profiles()
        self.fig.canvas.draw_idle()

    def plot_slices(self):
        # 使用第一个数据集作为背景图
        ref_name = self.names[0]
        data = self.data_cache[ref_name]
        
        # 1. Axial (Z=cz): Show X-Y (128x128)
        # 注意：imshow 默认 origin='upper' ((0,0)在左上)
        # 我们数据是 [Z, X, Y] -> slice [cz, :, :] -> X(行), Y(列)
        img_axial = data[self.cz, :, :]
        self.ax_axial.clear()
        self.ax_axial.imshow(img_axial, cmap='gray', aspect='equal', vmin=0, vmax=0.8)
        self.ax_axial.set_title(f"Axial (Z={self.cz})\nClick to move X/Y")
        self.ax_axial.axvline(self.cy, color='r', linestyle='--', alpha=0.5) # Y 是列
        self.ax_axial.axhline(self.cx, color='g', linestyle='--', alpha=0.5) # X 是行
        
        # 2. Lateral (Y=cy): Show Z-X (1024x128)
        # slice [:, :, cy] -> Z(行), X(列)
        # Aspect Ratio: Z_spacing(0.0362) / X_spacing(0.2) ≈ 0.18
        img_lat = data[:, :, self.cy]
        self.ax_lat.clear()
        ar_lat = self.spacing[0] / self.spacing[1]
        self.ax_lat.imshow(img_lat, cmap='gray', aspect=ar_lat, vmin=0, vmax=0.8)
        self.ax_lat.set_title(f"Lateral (Y={self.cy})\nClick to move Z/X")
        self.ax_lat.axvline(self.cx, color='g', linestyle='--', alpha=0.5) # X 是列
        self.ax_lat.axhline(self.cz, color='b', linestyle='--', alpha=0.5) # Z 是行
        
        # 3. Elevation (X=cx): Show Z-Y (1024x128)
        # slice [:, cx, :] -> Z(行), Y(列)
        img_ele = data[:, self.cx, :]
        self.ax_ele.clear()
        ar_ele = self.spacing[0] / self.spacing[2]
        self.ax_ele.imshow(img_ele, cmap='gray', aspect=ar_ele, vmin=0, vmax=0.8)
        self.ax_ele.set_title(f"Elevation (X={self.cx})\nClick to move Z/Y")
        self.ax_ele.axvline(self.cy, color='r', linestyle='--', alpha=0.5) # Y 是列
        self.ax_ele.axhline(self.cz, color='b', linestyle='--', alpha=0.5) # Z 是行

    def plot_profiles(self):
        self.ax_prof_z.clear(); self.ax_prof_x.clear(); self.ax_prof_y.clear()
        metrics_buffer = []
        
        for name in self.names:
            data = self.data_cache[name]
            
            # 提取
            prof_z = data[:, self.cx, self.cy] # 沿深度
            prof_x = data[self.cz, :, self.cy] # 沿宽度
            prof_y = data[self.cz, self.cx, :] # 沿厚度
            
            # 计算
            fw_z, _, psl_z, ax_z, cv_z = compute_metrics(prof_z, self.spacing[0])
            fw_x, _, psl_x, ax_x, cv_x = compute_metrics(prof_x, self.spacing[1])
            fw_y, _, psl_y, ax_y, cv_y = compute_metrics(prof_y, self.spacing[2])
            
            lbl = f"{name}\nFWHM:{fw_z:.2f}/{fw_x:.2f}/{fw_y:.2f}"
            
            # 绘图 (居中)
            self.ax_prof_z.plot(ax_z - self.cz*self.spacing[0], cv_z, label=lbl)
            self.ax_prof_x.plot(ax_x - self.cx*self.spacing[1], cv_x)
            self.ax_prof_y.plot(ax_y - self.cy*self.spacing[2], cv_y)
            
            metrics_buffer.append([self.cz, self.cx, self.cy, name, 
                                   f"{fw_z:.3f}", f"{fw_x:.3f}", f"{fw_y:.3f}",
                                   f"{psl_z:.1f}", f"{psl_x:.1f}", f"{psl_y:.1f}"])

        # 装饰
        for ax, title in zip([self.ax_prof_z, self.ax_prof_x, self.ax_prof_y], 
                             ['Axial Profile (Z)', 'Lateral Profile (X)', 'Elev. Profile (Y)']):
            ax.set_title(title)
            ax.set_ylim(-60, 5)
            ax.grid(True, linestyle='--', alpha=0.5)
            ax.axhline(-6, color='r', linestyle=':', alpha=0.3)
        
        self.ax_prof_z.legend(fontsize=6, loc='lower center')
        
        # 存盘
        with open(OUTPUT_CSV, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(metrics_buffer)
        print(f"  -> 指标已保存 (Z={self.cz}, X={self.cx}, Y={self.cy})")

    def on_click(self, event):
        if event.inaxes is None: return
        
        # 坐标转换：Matplotlib 坐标系 -> 数组索引
        # event.xdata 是列索引 (Column)，event.ydata 是行索引 (Row)
        col = int(event.xdata + 0.5)
        row = int(event.ydata + 0.5)
        
        # 边界检查
        if col < 0 or row < 0: return
        
        updated = False
        
        if event.inaxes == self.ax_axial: # Axial (X-Y) -> Row=X, Col=Y
            if row < 128 and col < 128:
                self.cx, self.cy = row, col
                updated = True
                
        elif event.inaxes == self.ax_lat: # Lat (Z-X) -> Row=Z, Col=X
            if row < 1024 and col < 128:
                self.cz, self.cx = row, col
                updated = True
                
        elif event.inaxes == self.ax_ele: # Ele (Z-Y) -> Row=Z, Col=Y
            if row < 1024 and col < 128:
                self.cz, self.cy = row, col
                updated = True
                
        if updated:
            print(f"🖱️ 点击跳转 -> [Z={self.cz}, X={self.cx}, Y={self.cy}]")
            self.update_all()

if __name__ == '__main__':
    # 确保保存目录存在
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    viewer = PSFViewer(file_dict, VOXEL_SPACING, INITIAL_POS)