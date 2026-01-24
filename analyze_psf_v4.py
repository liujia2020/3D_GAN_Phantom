import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.widgets import RadioButtons
from scipy.interpolate import make_interp_spline
import csv

# ==========================================
# 1. 配置区域
# ==========================================
# 物理间距 [Z(深度), X(宽度), Y(厚度)]
VOXEL_SPACING = np.array([0.0362, 0.2, 0.2]) 

# 初始切片位置
INITIAL_POS = [500, 64, 64] 

# 文件列表 (您的真实路径)
file_dict = {
    'Ground Truth':    '/home/liujia/g_linux/test/simu_stand_fixed_v2/Simu_0010_GT_SQ.nii', 
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
            if data.size > target_size: data = data[:target_size]
            else:
                temp = np.zeros(target_size, dtype=np.float32)
                temp[:data.size] = data
                data = temp
        
        return data.reshape((1024, 128, 128), order='F')
    except Exception as e:
        print(f"读取错误: {e}")
        return None

def compute_metrics(profile, spacing):
    """计算 FWHM, FWTM, PSL"""
    x = np.arange(len(profile))
    if len(profile) < 4: return 0,0,-100, x*spacing, profile
    
    x_new = np.linspace(0, len(profile)-1, len(profile)*10)
    try:
        spl = make_interp_spline(x, profile, k=3)
        y_smooth = spl(x_new)
    except:
        y_smooth = profile; x_new = x
        
    y_max = y_smooth.max()
    if y_max > 1e-9: y_smooth /= y_max
    else: return 0,0,-100, x_new*spacing, y_smooth-100
    
    y_db = 20 * np.log10(np.maximum(y_smooth, 1e-5))
    
    # FWHM (-6dB)
    mask_6 = y_db >= -6.0
    fwhm = (x_new[np.where(mask_6)[0][-1]] - x_new[np.where(mask_6)[0][0]]) * spacing if np.any(mask_6) else 0
    
    # FWTM (-20dB)
    mask_20 = y_db >= -20.0
    fwtm = (x_new[np.where(mask_20)[0][-1]] - x_new[np.where(mask_20)[0][0]]) * spacing if np.any(mask_20) else 0
    
    # PSL
    mask_main = y_db >= -10.0
    mask_side = ~mask_main
    psl = np.max(y_db[mask_side]) if np.any(mask_side) else -100
    
    return fwhm, fwtm, psl, x_new*spacing, y_db

class PSFViewerV4:
    def __init__(self, file_map, voxel_spacing, init_pos):
        self.file_map = file_map
        self.spacing = voxel_spacing
        self.cz, self.cx, self.cy = init_pos
        self.data_cache = {}
        self.names = list(file_map.keys())
        self.current_model = self.names[0] # 默认显示第一个
        
        # 1. 加载数据
        print(f"🚀 初始化... 默认中心 [Z={self.cz}, X={self.cx}, Y={self.cy}]")
        for name, path in file_map.items():
            print(f"  -> 读取 {name}...")
            data = robust_load_nii(path)
            if data is not None:
                # 99.9% Robust Scaling
                vmax = np.percentile(data, 99.9)
                if vmax > 0: data = data / vmax
                self.data_cache[name] = data
        
        if not self.data_cache: raise RuntimeError("无有效数据")
        
        # 2. 创建界面布局
        self.fig = plt.figure(figsize=(18, 12))
        
        # 定义网格: 左侧留给按钮，右侧显示图
        gs = GridSpec(2, 4, width_ratios=[0.5, 1, 1, 1], height_ratios=[1.2, 1])
        
        # --- 左侧控制区 ---
        ax_radio = self.fig.add_subplot(gs[0, 0])
        ax_radio.set_title("Select Image View", fontsize=10, fontweight='bold')
        self.radio = RadioButtons(ax_radio, self.names, active=0)
        self.radio.on_clicked(self.change_model_view)
        
        # --- 上排：三视图 (动态切换) ---
        self.ax_axial = self.fig.add_subplot(gs[0, 1]) # Z-plane
        self.ax_lat   = self.fig.add_subplot(gs[0, 2]) # Y-plane
        self.ax_ele   = self.fig.add_subplot(gs[0, 3]) # X-plane
        
        # --- 下排：三曲线 (永远全显) ---
        self.ax_prof_z = self.fig.add_subplot(gs[1, 1:]) # 合并显示会更宽，但我还是分开吧
        # 为了对齐，还是分三个
        self.ax_prof_z = self.fig.add_subplot(gs[1, 1])
        self.ax_prof_x = self.fig.add_subplot(gs[1, 2])
        self.ax_prof_y = self.fig.add_subplot(gs[1, 3])
        
        # 连接点击事件
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        
        # 3. 初始绘制
        self.update_all()
        
        # CSV Header
        if not os.path.exists(OUTPUT_CSV):
            with open(OUTPUT_CSV, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Z_idx','X_idx','Y_idx','Model','FWHM_Z','FWHM_X','FWHM_Y','PSL_Z','PSL_X','PSL_Y'])
        
        print("\n✅ 界面就绪！\n  - 左侧单选框切换显示的图像\n  - 点击图像更新所有指标")
        plt.show()

    def change_model_view(self, label):
        """单选按钮回调：只更新上排图像"""
        self.current_model = label
        print(f"📺 切换视图至: {label}")
        self.plot_slices() # 只重画切片，不重算曲线（除非坐标变了）
        self.fig.canvas.draw_idle()

    def update_all(self):
        self.plot_slices()
        self.plot_profiles()
        self.fig.canvas.draw_idle()

    def plot_slices(self):
        """画上排三视图 (只画 current_model)"""
        data = self.data_cache[self.current_model]
        
        # 1. Axial (Z=cz) -> X-Y
        img_axial = data[self.cz, :, :]
        self.ax_axial.clear()
        self.ax_axial.imshow(img_axial, cmap='gray', aspect='equal', vmin=0, vmax=0.8)
        self.ax_axial.set_title(f"Axial (Z={self.cz})\nModel: {self.current_model}", fontsize=9)
        self.ax_axial.axvline(self.cy, color='r', linestyle='--')
        self.ax_axial.axhline(self.cx, color='g', linestyle='--')
        
        # 2. Lateral (Y=cy) -> Z-X
        img_lat = data[:, :, self.cy]
        self.ax_lat.clear()
        ar_lat = self.spacing[0] / self.spacing[1]
        self.ax_lat.imshow(img_lat, cmap='gray', aspect=ar_lat, vmin=0, vmax=0.8)
        self.ax_lat.set_title(f"Lateral (Y={self.cy})", fontsize=9)
        self.ax_lat.axvline(self.cx, color='g', linestyle='--')
        self.ax_lat.axhline(self.cz, color='b', linestyle='--')
        
        # 3. Elevation (X=cx) -> Z-Y
        img_ele = data[:, self.cx, :]
        self.ax_ele.clear()
        ar_ele = self.spacing[0] / self.spacing[2]
        self.ax_ele.imshow(img_ele, cmap='gray', aspect=ar_ele, vmin=0, vmax=0.8)
        self.ax_ele.set_title(f"Elevation (X={self.cx})", fontsize=9)
        self.ax_ele.axvline(self.cy, color='r', linestyle='--')
        self.ax_ele.axhline(self.cz, color='b', linestyle='--')

    def plot_profiles(self):
        """画下排曲线 (遍历所有 model，叠加显示)"""
        self.ax_prof_z.clear(); self.ax_prof_x.clear(); self.ax_prof_y.clear()
        metrics_buffer = []
        
        colors = ['k', 'r', 'g', 'b', 'm', 'c'] # 预定义一些颜色
        
        for idx, name in enumerate(self.names):
            data = self.data_cache[name]
            color = colors[idx % len(colors)]
            lw = 2 if name == self.current_model else 1 # 当前选中的模型线粗一点
            alpha = 1.0 if name == self.current_model else 0.7
            
            # 提取
            prof_z = data[:, self.cx, self.cy]
            prof_x = data[self.cz, :, self.cy]
            prof_y = data[self.cz, self.cx, :]
            
            # 计算
            fw_z, _, psl_z, ax_z, cv_z = compute_metrics(prof_z, self.spacing[0])
            fw_x, _, psl_x, ax_x, cv_x = compute_metrics(prof_x, self.spacing[1])
            fw_y, _, psl_y, ax_y, cv_y = compute_metrics(prof_y, self.spacing[2])
            
            # 简化的 Label，防止图例太长
            # 只有当鼠标悬停或需要时才看详细数据，这里只标 FWHM
            lbl = f"{name} (FWHM:{fw_x:.2f})" 
            
            # 绘图 (中心化显示)
            self.ax_prof_z.plot(ax_z - self.cz*self.spacing[0], cv_z, label=lbl, color=color, linewidth=lw, alpha=alpha)
            self.ax_prof_x.plot(ax_x - self.cx*self.spacing[1], cv_x, color=color, linewidth=lw, alpha=alpha)
            self.ax_prof_y.plot(ax_y - self.cy*self.spacing[2], cv_y, color=color, linewidth=lw, alpha=alpha)
            
            metrics_buffer.append([self.cz, self.cx, self.cy, name, 
                                   f"{fw_z:.3f}", f"{fw_x:.3f}", f"{fw_y:.3f}",
                                   f"{psl_z:.1f}", f"{psl_x:.1f}", f"{psl_y:.1f}"])

        # 装饰
        for ax, title in zip([self.ax_prof_z, self.ax_prof_x, self.ax_prof_y], 
                             ['Axial (Z)', 'Lateral (X)', 'Elevation (Y)']):
            ax.set_title(title, fontsize=9)
            ax.set_ylim(-60, 5)
            ax.grid(True, linestyle='--', alpha=0.5)
            ax.axhline(-6, color='gray', linestyle=':', linewidth=0.8)
        
        # 只在第一张图显示图例，避免拥挤
        self.ax_prof_z.legend(fontsize=7, loc='lower center', framealpha=0.8)
        
        # 存盘
        with open(OUTPUT_CSV, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(metrics_buffer)
        print(f"  -> 指标已保存")

    def on_click(self, event):
        if event.inaxes not in [self.ax_axial, self.ax_lat, self.ax_ele]: return
        
        col = int(event.xdata + 0.5)
        row = int(event.ydata + 0.5)
        
        if col < 0 or row < 0: return
        updated = False
        
        if event.inaxes == self.ax_axial: # Axial (X-Y) -> Row=X, Col=Y
            if row < 128 and col < 128: self.cx, self.cy = row, col; updated = True
        elif event.inaxes == self.ax_lat: # Lat (Z-X) -> Row=Z, Col=X
            if row < 1024 and col < 128: self.cz, self.cx = row, col; updated = True
        elif event.inaxes == self.ax_ele: # Ele (Z-Y) -> Row=Z, Col=Y
            if row < 1024 and col < 128: self.cz, self.cy = row, col; updated = True
                
        if updated:
            print(f"🖱️  -> 跳转 [Z={self.cz}, X={self.cx}, Y={self.cy}]")
            self.update_all()

if __name__ == '__main__':
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    viewer = PSFViewerV4(file_dict, VOXEL_SPACING, INITIAL_POS)