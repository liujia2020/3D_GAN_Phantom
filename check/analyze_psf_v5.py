import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
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

# 文件列表
file_dict = {
    'Ground Truth':    '/home/liujia/g_linux/test/simu_stand_fixed_v2/Simu_0010_GT_SQ.nii', 
    
    'Exp1 (Pixel100)': '/mnt/g/train_data/results/06_1Ch_L1_Only/Simu_001_06_1Ch_L1_Only_Fake.nii',
    'Exp2 (Pixel100)': '/mnt/g/train_data/results/07_1Ch_Standard_GAN/Simu_001_07_1Ch_Standard_GAN_Fake.nii',
    'Exp3 (Pixel100)': '/mnt/g/train_data/results/08_1Ch_Strong_GAN/Simu_001_08_1Ch_Strong_GAN_Fake.nii',
    'Exp4 (Pixel100)': '/mnt/g/train_data/results/09_1Ch_Smooth_TV/Simu_001_09_1Ch_Smooth_TV_Fake.nii',
    'Exp5 (Pixel100)': '/mnt/g/train_data/results/10_1Ch_StrongGAN_Edge/Simu_001_10_1Ch_StrongGAN_Edge_Fake.nii',
    'Exp6 (Pixel100)': '/mnt/g/train_data/results/11_1Ch_StrongGAN_Percep/Simu_001_11_1Ch_StrongGAN_Percep_Fake.nii',
    'Exp7 (Pixel100)': '/mnt/g/train_data/results/12_1Ch_Hybrid_EdgePercep/Simu_001_12_1Ch_Hybrid_EdgePercep_Fake.nii',
    'Exp8 (Pixel100)': '/mnt/g/train_data/results/13_1Ch_UltraGAN_TV/Simu_001_13_1Ch_UltraGAN_TV_Fake.nii',
    'Exp9 (Pixel100)': '/mnt/g/train_data/results/14_1Ch_Composite_Geo/Simu_001_14_1Ch_Composite_Geo_Fake.nii',
}

OUTPUT_CSV = '/mnt/g/result_ana/psf_metrics_analysis.csv'

# ==========================================
# 2. 样式配置 (针对大屏优化)
# ==========================================
plt.rcParams.update({
    'font.size': 14,          # 全局基础字体
    'axes.titlesize': 20,     # 标题字体
    'axes.labelsize': 16,     # 坐标轴标签字体
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'lines.linewidth': 2.5,   # 全局线宽
    'figure.titlesize': 24
})

# ==========================================
# 3. 核心逻辑
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

# ======= [修改 1] 替换整个 compute_metrics 函数 =======
def compute_metrics(profile, spacing):
    """计算 FWHM, FWTM, PSL (兼容 dB 输入)"""
    x = np.arange(len(profile))
    if len(profile) < 4: return 0,0,-100, x*spacing, profile
    
    x_new = np.linspace(0, len(profile)-1, len(profile)*10)
    try:
        spl = make_interp_spline(x, profile, k=3)
        y_smooth = spl(x_new)
    except:
        y_smooth = profile; x_new = x
    
    # 核心修改：不再做 log，而是将峰值对齐到 0dB
    y_max = np.max(y_smooth)
    y_db = y_smooth - y_max # 峰值归一化为 0
    
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
# ====================================================
class PSFViewerV5:
    def __init__(self, file_map, voxel_spacing, init_pos):
        self.file_map = file_map
        self.spacing = voxel_spacing
        self.cz, self.cx, self.cy = init_pos
        self.data_cache = {}
        self.names = list(file_map.keys())
        self.current_model = self.names[0] 
        
        # # 1. 加载数据
        # print(f"🚀 初始化... 默认中心 [Z={self.cz}, X={self.cx}, Y={self.cy}]")
        # for name, path in file_map.items():
        #     print(f"  -> 读取 {name}...")
        #     data = robust_load_nii(path)
        #     if data is not None:
        #         vmax = np.percentile(data, 99.9)
        #         if vmax > 0: data = data / vmax
        #         self.data_cache[name] = data
        
        # if not self.data_cache: raise RuntimeError("无有效数据")
        # ======= [修改 2] 替换 __init__ 中的加载循环 =======
        # 1. 加载数据
        print(f"🚀 初始化... 默认中心 [Z={self.cz}, X={self.cx}, Y={self.cy}]")
        for name, path in file_map.items():
            print(f"  -> 读取 {name}...")
            data = robust_load_nii(path)
            if data is not None:
                # --- 智能数据转换逻辑 ---
                max_val = np.max(data)
                if max_val > 0: 
                    # 情况A: 线性数据 (如 GT) -> 执行 Log 变换
                    data = np.abs(data)
                    data = data / (np.max(data) + 1e-9) + 1e-9
                    data = 20 * np.log10(data)
                else:
                    # 情况B: dB 数据 (如 Exp1) -> 保持原样
                    pass

                # 统一截断到 -60dB 并将峰值对齐到 0
                data = np.clip(data, -60, 0)
                if np.max(data) > -60:
                    data = data - np.max(data)

                self.data_cache[name] = data
        
        if not self.data_cache: raise RuntimeError("无有效数据")
        # =================================================       
        # 2. 创建大屏界面布局 (Nested GridSpec)
        # figsize 设为 (20, 12) 以适应大屏
        self.fig = plt.figure(figsize=(20, 12), constrained_layout=True)
        
        # 主网格：上下两行，高度比例 1.2 : 1
        gs_main = gridspec.GridSpec(2, 1, figure=self.fig, height_ratios=[1.2, 1])
        
        # --- 上半区网格 (1行4列) ---
        # 宽比：0.5 (控制区) : 1 : 1 : 1 (三视图)
        gs_top = gridspec.GridSpecFromSubplotSpec(1, 4, subplot_spec=gs_main[0], width_ratios=[0.5, 1, 1, 1])
        
        # --- 下半区网格 (1行3列) ---
        # 宽比：1 : 1 : 1 (三曲线撑满全屏)
        gs_bot = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=gs_main[1])
        
        # 组件初始化
        # 左侧控制区
        ax_radio = self.fig.add_subplot(gs_top[0, 0])
        ax_radio.set_title("Select View", fontsize=22, fontweight='bold', pad=20)
        self.radio = RadioButtons(ax_radio, self.names, active=0)
        
        # 暴力放大按钮字体
        for label in self.radio.labels:
            label.set_fontsize(18)
            label.set_fontweight('bold')
        
        self.radio.on_clicked(self.change_model_view)
        
        # 上排三视图
        self.ax_axial = self.fig.add_subplot(gs_top[0, 1])
        self.ax_lat   = self.fig.add_subplot(gs_top[0, 2])
        self.ax_ele   = self.fig.add_subplot(gs_top[0, 3])
        
        # 下排三曲线
        self.ax_prof_z = self.fig.add_subplot(gs_bot[0, 0])
        self.ax_prof_x = self.fig.add_subplot(gs_bot[0, 1])
        self.ax_prof_y = self.fig.add_subplot(gs_bot[0, 2])
        
        # 连接点击事件
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        
        # 3. 初始绘制
        self.update_all()
        
        # CSV Header
        if not os.path.exists(OUTPUT_CSV):
            with open(OUTPUT_CSV, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Z_idx','X_idx','Y_idx','Model','FWHM_Z','FWHM_X','FWHM_Y','PSL_Z','PSL_X','PSL_Y'])
        
        print("\n✅ V5 全屏版界面就绪！")
        plt.show()

    def change_model_view(self, label):
        self.current_model = label
        self.plot_slices()
        self.fig.canvas.draw_idle()

    def update_all(self):
        self.plot_slices()
        self.plot_profiles()
        self.fig.canvas.draw_idle()

    def plot_slices(self):
        """画上排三视图"""
        data = self.data_cache[self.current_model]
        
        # 十字光标样式
        cross_style = {'color': 'red', 'linestyle': '--', 'linewidth': 2.0, 'alpha': 0.8}
        blue_style = {'color': 'cyan', 'linestyle': '--', 'linewidth': 2.0, 'alpha': 0.8} # 区分Z轴
        
        # 1. Axial (Z=cz) -> X-Y
        img_axial = data[self.cz, :, :]
        self.ax_axial.clear()
        # self.ax_axial.imshow(img_axial, cmap='gray', aspect='equal', vmin=0, vmax=0.8)
        self.ax_axial.imshow(img_axial, cmap='gray', aspect='equal', vmin=-60, vmax=0)
        self.ax_axial.set_title(f"Axial (Z={self.cz})\n{self.current_model}", fontweight='bold')
        self.ax_axial.axvline(self.cy, **cross_style)
        self.ax_axial.axhline(self.cx, **cross_style)
        
        # 2. Lateral (Y=cy) -> Z-X
        img_lat = data[:, :, self.cy]
        self.ax_lat.clear()
        ar_lat = self.spacing[0] / self.spacing[1]
        # self.ax_lat.imshow(img_lat, cmap='gray', aspect=ar_lat, vmin=0, vmax=0.8)
        self.ax_lat.imshow(img_lat, cmap='gray', aspect=ar_lat, vmin=-60, vmax=0)
        self.ax_lat.set_title(f"Lateral (Y={self.cy})", fontweight='bold')
        self.ax_lat.axvline(self.cx, **cross_style)
        self.ax_lat.axhline(self.cz, **blue_style)
        
        # 3. Elevation (X=cx) -> Z-Y
        img_ele = data[:, self.cx, :]
        self.ax_ele.clear()
        ar_ele = self.spacing[0] / self.spacing[2]
        # self.ax_ele.imshow(img_ele, cmap='gray', aspect=ar_ele, vmin=0, vmax=0.8)
        self.ax_ele.imshow(img_ele, cmap='gray', aspect=ar_ele, vmin=-60, vmax=0)
        self.ax_ele.set_title(f"Elevation (X={self.cx})", fontweight='bold')
        self.ax_ele.axvline(self.cy, **cross_style)
        self.ax_ele.axhline(self.cz, **blue_style)

    def plot_profiles(self):
        """画下排曲线"""
        self.ax_prof_z.clear(); self.ax_prof_x.clear(); self.ax_prof_y.clear()
        metrics_buffer = []
        
        colors = ['k', 'r', 'g', 'b', 'm', 'c'] 
        
        for idx, name in enumerate(self.names):
            data = self.data_cache[name]
            color = colors[idx % len(colors)]
            
            # 选中模型加粗
            is_active = (name == self.current_model)
            lw = 4.0 if is_active else 2.0 
            alpha = 1.0 if is_active else 0.6
            z_order = 10 if is_active else 1
            
            # 提取
            prof_z = data[:, self.cx, self.cy]
            prof_x = data[self.cz, :, self.cy]
            prof_y = data[self.cz, self.cx, :]
            
            # 计算
            fw_z, _, psl_z, ax_z, cv_z = compute_metrics(prof_z, self.spacing[0])
            fw_x, _, psl_x, ax_x, cv_x = compute_metrics(prof_x, self.spacing[1])
            fw_y, _, psl_y, ax_y, cv_y = compute_metrics(prof_y, self.spacing[2])
            
            lbl = f"{name}" if idx==0 else None # 仅在第一个图显示图例，或按需显示
            
            # 绘图
            self.ax_prof_z.plot(ax_z - self.cz*self.spacing[0], cv_z, label=name, color=color, linewidth=lw, alpha=alpha, zorder=z_order)
            self.ax_prof_x.plot(ax_x - self.cx*self.spacing[1], cv_x, color=color, linewidth=lw, alpha=alpha, zorder=z_order)
            self.ax_prof_y.plot(ax_y - self.cy*self.spacing[2], cv_y, color=color, linewidth=lw, alpha=alpha, zorder=z_order)
            
            metrics_buffer.append([self.cz, self.cx, self.cy, name, 
                                   f"{fw_z:.3f}", f"{fw_x:.3f}", f"{fw_y:.3f}",
                                   f"{psl_z:.1f}", f"{psl_x:.1f}", f"{psl_y:.1f}"])

        # 装饰
        for ax, title, tag in zip([self.ax_prof_z, self.ax_prof_x, self.ax_prof_y], 
                             ['Axial Profile (Z)', 'Lateral Profile (X)', 'Elevation Profile (Y)'],
                             ['mm', 'mm', 'mm']):
            ax.set_title(title, fontweight='bold', pad=15)
            ax.set_xlabel(f"Distance ({tag})")
            ax.set_ylim(-60, 5)
            ax.grid(True, linestyle='--', alpha=0.5, linewidth=1.5)
            ax.axhline(-6, color='gray', linestyle=':', linewidth=2.0)
            
            # 标记 FWHM 宽度
            ax.text(0.05, 0.9, "-6dB Width", transform=ax.transAxes, fontsize=12, color='gray')

        # 图例只放在最左边的图，字体加大
        self.ax_prof_z.legend(fontsize=12, loc='lower center', framealpha=0.9, facecolor='white')
        
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
        
        if event.inaxes == self.ax_axial: 
            if row < 128 and col < 128: self.cx, self.cy = row, col; updated = True
        elif event.inaxes == self.ax_lat: 
            if row < 1024 and col < 128: self.cz, self.cx = row, col; updated = True
        elif event.inaxes == self.ax_ele: 
            if row < 1024 and col < 128: self.cz, self.cy = row, col; updated = True
                
        if updated:
            print(f"🖱️  -> 跳转 [Z={self.cz}, X={self.cx}, Y={self.cy}]")
            self.update_all()

if __name__ == '__main__':
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    viewer = PSFViewerV5(file_dict, VOXEL_SPACING, INITIAL_POS)