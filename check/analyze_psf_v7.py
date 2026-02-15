import os
import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use('TkAgg') # 使用窗口显示
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.widgets import RadioButtons, Button
from scipy.interpolate import make_interp_spline
import csv

# ==========================================
# 1. 全局配置区域 (暴力大字体优化)
# ==========================================
# 物理间距 [Z(深度), X(宽度), Y(厚度)]
VOXEL_SPACING = np.array([0.0362, 0.2, 0.2]) 

# 初始切片位置
INITIAL_POS = [500, 64, 64] 

# 文件列表
file_dict = {
    'Ground Truth':    '/home/liujia/g_linux/test/simu_stand_fixed_v2/Simu_0010_GT_SQ.nii', 
    
    # 'Exp1 (Pixel100)': '/mnt/g/train_data/results/06_1Ch_L1_Only/Simu_001_06_1Ch_L1_Only_Fake.nii',
    # 'Exp2 (Pixel100)': '/mnt/g/train_data/results/07_1Ch_Standard_GAN/Simu_001_07_1Ch_Standard_GAN_Fake.nii',
    # 'Exp3 (Pixel100)': '/mnt/g/train_data/results/08_1Ch_Strong_GAN/Simu_001_08_1Ch_Strong_GAN_Fake.nii',
    # 'Exp4 (Pixel100)': '/mnt/g/train_data/results/09_1Ch_Smooth_TV/Simu_001_09_1Ch_Smooth_TV_Fake.nii',
    # 'Exp5 (Pixel100)': '/mnt/g/train_data/results/10_1Ch_StrongGAN_Edge/Simu_001_10_1Ch_StrongGAN_Edge_Fake.nii',
    # 'Exp6 (Pixel100)': '/mnt/g/train_data/results/11_1Ch_StrongGAN_Percep/Simu_001_11_1Ch_StrongGAN_Percep_Fake.nii',
    # 'Exp7 (Pixel100)': '/mnt/g/train_data/results/12_1Ch_Hybrid_EdgePercep/Simu_001_12_1Ch_Hybrid_EdgePercep_Fake.nii',
    # 'Exp8 (Pixel100)': '/mnt/g/train_data/results/13_1Ch_UltraGAN_TV/Simu_001_13_1Ch_UltraGAN_TV_Fake.nii',
    # 'Exp9 (Pixel100)': '/mnt/g/train_data/results/14_1Ch_Composite_Geo/Simu_001_14_1Ch_Composite_Geo_Fake.nii',
    'Exp17': '/mnt/g/train_data/results/17_GAN5_L1_100_Attention/Simu_001_17_GAN5_L1_100_Attention_Fake.nii',
    'Exp18': '/mnt/g/train_data/results/18_GAN5_L1_100_SSIM1_Attn/Simu_001_18_GAN5_L1_100_SSIM1_Attn_Fake.nii',
    'Exp19': '/mnt/g/train_data/results/19_GAN5_L1_100_Percep2_Attn/Simu_001_19_GAN5_L1_100_Percep2_Attn_Fake.nii',
    'Exp20': '/mnt/g/train_data/results/20_GAN5_L1_50_Attn/Simu_001_20_GAN5_L1_50_Attn_Fake.nii',
    'Exp21': '/mnt/g/train_data/results/21_GAN10_L1_100_Attn/Simu_001_21_GAN10_L1_100_Attn_Fake.nii',
    'Exp22': '/mnt/g/train_data/results/22_GAN10_L1_100_SSIM1_Attn/Simu_001_22_GAN10_L1_100_SSIM1_Attn_Fake.nii',
    'Exp23': '/mnt/g/train_data/results/23_GAN5_L1_20_SSIM1_Attn/Simu_001_23_GAN5_L1_20_SSIM1_Attn_Fake.nii',

}

OUTPUT_CSV = '/mnt/g/result_ana/psf_metrics_analysis_v7.csv'

# --- Matplotlib 全局字体暴力放大 ---
plt.rcParams.update({
    'font.size': 14,          # 基础字体
    'axes.titlesize': 16,     # 标题字体
    'axes.labelsize': 14,     # 轴标签
    'xtick.labelsize': 12,    # 刻度
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'figure.titlesize': 20
})

# ==========================================
# 2. 核心辅助函数 (保持不变)
# ==========================================

def robust_load_nii(path):
    """智能加载: 自动识别 Linear/dB, 统一转为 [-60, 0] dB"""
    if not os.path.exists(path):
        print(f"❌ 文件不存在: {path}")
        return None

    try:
        nii = nib.load(path)
        data = nii.get_fdata().astype(np.float32)
    except:
        print(f"⚠️ 标准读取失败，尝试二进制流读取: {path}")
        try:
            with open(path, 'rb') as f:
                f.seek(352)
                raw = f.read()
            data = np.frombuffer(raw, dtype=np.float32)
            data = data.reshape((1024, 128, 128), order='F')
        except Exception as e:
            print(f"❌ 读取彻底失败: {e}")
            return None

    max_val = np.max(data)
    if max_val > 0:
        data = np.abs(data)
        data = data / (np.max(data) + 1e-9) + 1e-9
        data = 20 * np.log10(data)
    
    data = np.clip(data, -60, 0)
    curr_max = np.max(data)
    if curr_max > -60:
        data = data - curr_max

    return data

def compute_metrics(profile, spacing):
    """计算指标 (兼容 dB 输入)"""
    x = np.arange(len(profile))
    if len(profile) < 4: return 0,0,-100, x*spacing, profile
    
    x_new = np.linspace(0, len(profile)-1, len(profile)*10)
    try:
        spl = make_interp_spline(x, profile, k=3)
        y_smooth = spl(x_new)
    except:
        y_smooth = profile; x_new = x
    
    y_max = np.max(y_smooth)
    y_db = y_smooth - y_max 
    
    mask_6 = y_db >= -6.0
    fwhm = (x_new[np.where(mask_6)[0][-1]] - x_new[np.where(mask_6)[0][0]]) * spacing if np.any(mask_6) else 0
    
    mask_20 = y_db >= -20.0
    fwtm = (x_new[np.where(mask_20)[0][-1]] - x_new[np.where(mask_20)[0][0]]) * spacing if np.any(mask_20) else 0
    
    mask_main = y_db >= -10.0
    mask_side = ~mask_main
    psl = np.max(y_db[mask_side]) if np.any(mask_side) else -100
    
    return fwhm, fwtm, psl, x_new*spacing, y_db

# ==========================================
# 3. PSFViewerV7 主类 (布局重构)
# ==========================================

class PSFViewerV7:
    def __init__(self, file_map, voxel_spacing, init_pos):
        self.file_map = file_map
        self.spacing = voxel_spacing
        self.cz, self.cx, self.cy = init_pos
        self.data_cache = {}
        self.names = list(file_map.keys())
        self.show_all_mode = False 
        
        # 寻找 GT
        self.gt_name = 'Ground Truth' if 'Ground Truth' in self.names else self.names[0]
        
        # 默认选中第一个非 GT 模型
        self.current_model = self.gt_name
        for n in self.names:
            if n != self.gt_name:
                self.current_model = n
                break
        
        # 加载数据
        print(f"🚀 初始化 V7... GT is '{self.gt_name}'")
        for name, path in file_map.items():
            print(f"  -> 读取 {name}...")
            data = robust_load_nii(path)
            if data is not None:
                self.data_cache[name] = data
        
        if not self.data_cache: raise RuntimeError("无有效数据")
        
        # --- 布局重构: 20x15 大屏, 极限边距 ---
        self.fig = plt.figure(figsize=(20, 15))
        
        # 1. 定义主网格: 左(10%) vs 右(90%) -> 极限压缩左侧
        # left=0.01, right=0.99 (贴边显示)
        gs_main = gridspec.GridSpec(1, 2, figure=self.fig, 
                                    width_ratios=[0.1, 0.9], 
                                    wspace=0.02, 
                                    left=0.01, right=0.99, top=0.98, bottom=0.02)
        
        # --- 左侧控制区 ---
        gs_left = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs_main[0], height_ratios=[4, 1], hspace=0.05)
        
        # Radio Buttons
        self.ax_radio = self.fig.add_subplot(gs_left[0])
        self.ax_radio.set_title("Select Exp", fontsize=18, fontweight='bold', pad=10)
        self.ax_radio.set_facecolor('#f5f5f5')
        
        self.exp_names = [n for n in self.names if n != self.gt_name]
        if not self.exp_names: self.exp_names = [self.gt_name]
            
        self.radio = RadioButtons(self.ax_radio, self.exp_names, active=0)
        self.radio.on_clicked(self.change_model_view)
        
        # [暴力修改] Radio Button 字体大小
        for label in self.radio.labels:
            label.set_fontsize(16)      # 列表字体变大
            label.set_fontweight('bold')
        # # 加大圆圈的大小
        # for circle in self.radio.circles:
        #     circle.set_radius(0.05)
        # [修正] 尝试加大圆圈 (兼容性处理)
        try:
            # 某些 Matplotlib 版本可能没有 circles 属性，这里做个保护
            if hasattr(self.radio, 'circles'):
                for circle in self.radio.circles:
                    circle.set_radius(0.05)
        except AttributeError:
            pass
        # Mode Button
        self.ax_btn = self.fig.add_subplot(gs_left[1])
        self.btn = Button(self.ax_btn, 'Mode:\nCompare', color='lightblue', hovercolor='skyblue')
        self.btn.label.set_fontsize(18) # 按钮字体变大
        self.btn.label.set_fontweight('bold')
        self.btn.on_clicked(self.toggle_mode)
        
        # --- 右侧显示区 (三行) ---
        # 压缩行间距 hspace=0.15
        gs_right = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=gs_main[1], 
                                                    height_ratios=[1, 1, 0.8], 
                                                    hspace=0.15)
        
        # 第 1 行: GT Views (Fixed) - 压缩子图间距 wspace=0.02
        gs_row1 = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=gs_right[0], wspace=0.02)
        self.ax_gt_ax  = self.fig.add_subplot(gs_row1[0])
        self.ax_gt_lat = self.fig.add_subplot(gs_row1[1])
        self.ax_gt_ele = self.fig.add_subplot(gs_row1[2])
        
        # 第 2 行: Exp Views (Dynamic)
        gs_row2 = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=gs_right[1], wspace=0.02)
        self.ax_exp_ax  = self.fig.add_subplot(gs_row2[0])
        self.ax_exp_lat = self.fig.add_subplot(gs_row2[1])
        self.ax_exp_ele = self.fig.add_subplot(gs_row2[2])
        
        # 第 3 行: Profiles (稍微宽一点的间距 wspace=0.10 用于显示Y轴文字)
        gs_row3 = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=gs_right[2], wspace=0.10)
        self.ax_prof_z = self.fig.add_subplot(gs_row3[0])
        self.ax_prof_x = self.fig.add_subplot(gs_row3[1])
        self.ax_prof_y = self.fig.add_subplot(gs_row3[2])
        
        self.slice_axes = [
            self.ax_gt_ax, self.ax_gt_lat, self.ax_gt_ele,
            self.ax_exp_ax, self.ax_exp_lat, self.ax_exp_ele
        ]

        # 连接事件
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        
        # 初始化 CSV
        if not os.path.exists(OUTPUT_CSV):
            with open(OUTPUT_CSV, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Z_idx','X_idx','Y_idx','Model','FWHM_Z','FWHM_X','FWHM_Y','PSL_Z','PSL_X','PSL_Y'])
        
        self.update_all()
        print("\n✅ V7 极限大屏版就绪！请最大化窗口体验。")
        plt.show()

    # ==========================================
    # 交互回调
    # ==========================================
    
    def change_model_view(self, label):
        self.current_model = label
        self.update_all()

    def toggle_mode(self, event):
        self.show_all_mode = not self.show_all_mode
        self.btn.label.set_text('Mode:\nShow All' if self.show_all_mode else 'Mode:\nCompare')
        self.plot_profiles()
        self.fig.canvas.draw_idle()

    def on_click(self, event):
        if event.inaxes not in self.slice_axes: return
        
        col = int(event.xdata + 0.5)
        row = int(event.ydata + 0.5)
        updated = False
        
        # Axial (XY) -> row=X, col=Y
        if event.inaxes in [self.ax_gt_ax, self.ax_exp_ax]:
            if row < 128 and col < 128:
                self.cx, self.cy = row, col; updated = True
        # Lateral (ZX) -> row=Z, col=X
        elif event.inaxes in [self.ax_gt_lat, self.ax_exp_lat]:
            if row < 1024 and col < 128:
                self.cz, self.cx = row, col; updated = True
        # Elevation (ZY) -> row=Z, col=Y
        elif event.inaxes in [self.ax_gt_ele, self.ax_exp_ele]:
            if row < 1024 and col < 128:
                self.cz, self.cy = row, col; updated = True
        
        if updated:
            print(f"🖱️  -> Jump to [Z={self.cz}, X={self.cx}, Y={self.cy}]")
            self.update_all()

    # ==========================================
    # 绘图逻辑
    # ==========================================
    
    def update_all(self):
        # GT 行
        self.plot_slices_row(self.gt_name, 
                             [self.ax_gt_ax, self.ax_gt_lat, self.ax_gt_ele], 
                             title_prefix="[GT]")
        # Exp 行
        self.plot_slices_row(self.current_model, 
                             [self.ax_exp_ax, self.ax_exp_lat, self.ax_exp_ele], 
                             title_prefix="[Exp]")
        # Profiles
        self.plot_profiles()
        self.fig.canvas.draw_idle()

    def plot_slices_row(self, model_name, axes_list, title_prefix=""):
        ax_ax, ax_lat, ax_ele = axes_list
        data = self.data_cache.get(model_name)
        if data is None: return

        # 样式配置
        cross_style = {'color': 'red', 'linestyle': '--', 'linewidth': 1.5, 'alpha': 0.9}
        blue_style = {'color': 'cyan', 'linestyle': '--', 'linewidth': 1.5, 'alpha': 0.9}
        
        # Axial
        img_axial = data[self.cz, :, :]
        ax_ax.clear()
        ax_ax.imshow(img_axial, cmap='gray', aspect='equal', vmin=-60, vmax=0)
        ax_ax.set_title(f"{title_prefix} Axial (Z={self.cz})\n{model_name}", fontweight='bold')
        ax_ax.axvline(self.cy, **cross_style)
        ax_ax.axhline(self.cx, **cross_style)
        ax_ax.axis('off') # 关闭坐标轴省空间

        # Lateral
        img_lat = data[:, :, self.cy]
        ax_lat.clear()
        ar_lat = self.spacing[0] / self.spacing[1]
        ax_lat.imshow(img_lat, cmap='gray', aspect=ar_lat, vmin=-60, vmax=0)
        ax_lat.set_title(f"{title_prefix} Lateral (Y={self.cy})", fontweight='bold')
        ax_lat.axvline(self.cx, **cross_style)
        ax_lat.axhline(self.cz, **blue_style)
        ax_lat.axis('off')

        # Elevation
        img_ele = data[:, self.cx, :]
        ax_ele.clear()
        ar_ele = self.spacing[0] / self.spacing[2]
        ax_ele.imshow(img_ele, cmap='gray', aspect=ar_ele, vmin=-60, vmax=0)
        ax_ele.set_title(f"{title_prefix} Elevation (X={self.cx})", fontweight='bold')
        ax_ele.axvline(self.cy, **cross_style)
        ax_ele.axhline(self.cz, **blue_style)
        ax_ele.axis('off')

    def plot_profiles(self):
        self.ax_prof_z.clear(); self.ax_prof_x.clear(); self.ax_prof_y.clear()
        metrics_buffer = []
        
        target_models = self.names if self.show_all_mode else [self.gt_name, self.current_model]
        target_models = list(dict.fromkeys(target_models)) # 去重
        
        colors = ['k', 'r', 'g', 'b', 'm', 'c']
        
        for idx, name in enumerate(target_models):
            data = self.data_cache.get(name)
            if data is None: continue
            
            is_gt = (name == self.gt_name)
            is_selected = (name == self.current_model)
            
            if is_gt:
                color, ls, lw, zorder = 'k', '--', 3.0, 10
            elif is_selected:
                color, ls, lw, zorder = 'r', '-', 3.0, 9
            else:
                color, ls, lw, zorder = colors[idx % len(colors)], '-', 1.5, 1

            prof_z = data[:, self.cx, self.cy]
            prof_x = data[self.cz, :, self.cy]
            prof_y = data[self.cz, self.cx, :]
            
            fw_z, _, psl_z, ax_z, cv_z = compute_metrics(prof_z, self.spacing[0])
            fw_x, _, psl_x, ax_x, cv_x = compute_metrics(prof_x, self.spacing[1])
            fw_y, _, psl_y, ax_y, cv_y = compute_metrics(prof_y, self.spacing[2])
            
            self.ax_prof_z.plot(ax_z - self.cz*self.spacing[0], cv_z, label=name, color=color, linestyle=ls, linewidth=lw, zorder=zorder)
            self.ax_prof_x.plot(ax_x - self.cx*self.spacing[1], cv_x, color=color, linestyle=ls, linewidth=lw, zorder=zorder)
            self.ax_prof_y.plot(ax_y - self.cy*self.spacing[2], cv_y, color=color, linestyle=ls, linewidth=lw, zorder=zorder)
            
            if is_gt or is_selected:
                metrics_buffer.append([self.cz, self.cx, self.cy, name, 
                                       f"{fw_z:.3f}", f"{fw_x:.3f}", f"{fw_y:.3f}",
                                       f"{psl_z:.1f}", f"{psl_x:.1f}", f"{psl_y:.1f}"])

        for ax, title in zip([self.ax_prof_z, self.ax_prof_x, self.ax_prof_y], 
                             ['Axial Prof (Z)', 'Lateral Prof (X)', 'Elevation Prof (Y)']):
            ax.set_title(title, fontweight='bold')
            ax.set_ylim(-65, 5)
            ax.grid(True, linestyle='--', alpha=0.5)
            ax.axhline(-6, color='gray', linestyle=':', linewidth=1.5)

        self.ax_prof_z.legend(fontsize=12, loc='lower center', framealpha=0.9)
        
        if metrics_buffer:
            with open(OUTPUT_CSV, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerows(metrics_buffer)

if __name__ == '__main__':
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    viewer = PSFViewerV7(file_dict, VOXEL_SPACING, INITIAL_POS)