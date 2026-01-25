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

OUTPUT_CSV = '/mnt/g/result_ana/psf_metrics_analysis_v6.csv'

# ==========================================
# 2. 核心辅助函数
# ==========================================

def robust_load_nii(path):
    """
    智能加载 NIfTI 文件：
    - 自动识别是线性数据还是 dB 数据
    - 统一转换为 [-60, 0] dB 范围
    """
    if not os.path.exists(path):
        print(f"❌ 文件不存在: {path}")
        return None

    try:
        # 优先尝试标准 nibabel 读取
        nii = nib.load(path)
        data = nii.get_fdata().astype(np.float32)
    except:
        print(f"⚠️ 标准读取失败，尝试二进制流读取: {path}")
        try:
            with open(path, 'rb') as f:
                f.seek(352)
                raw = f.read()
            data = np.frombuffer(raw, dtype=np.float32)
            # 假设尺寸固定为 1024x128x128，根据实际情况调整
            data = data.reshape((1024, 128, 128), order='F')
        except Exception as e:
            print(f"❌ 读取彻底失败: {e}")
            return None

    # --- 智能数据范围判断 ---
    max_val = np.max(data)
    min_val = np.min(data)
    
    # print(f"  Debug: Range [{min_val:.2f}, {max_val:.2f}]")

    if max_val > 0:
        # 情况A: 线性数据 (Ground Truth) -> 执行 Log 变换
        data = np.abs(data)
        data = data / (np.max(data) + 1e-9) + 1e-9
        data = 20 * np.log10(data)
    else:
        # 情况B: dB 数据 (Exp Result) -> 已经是 dB 了，保持原样
        pass

    # 统一截断到 [-60, 0] dB
    data = np.clip(data, -60, 0)
    
    # 再次确保峰值在 0 (防止整体过暗)
    curr_max = np.max(data)
    if curr_max > -60:
        data = data - curr_max

    return data

def compute_metrics(profile, spacing):
    """计算 FWHM, FWTM, PSL (兼容 dB 输入)"""
    x = np.arange(len(profile))
    if len(profile) < 4: return 0,0,-100, x*spacing, profile
    
    # 插值平滑
    x_new = np.linspace(0, len(profile)-1, len(profile)*10)
    try:
        spl = make_interp_spline(x, profile, k=3)
        y_smooth = spl(x_new)
    except:
        y_smooth = profile; x_new = x
    
    # 归一化到 0dB
    y_max = np.max(y_smooth)
    y_db = y_smooth - y_max 
    
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

# ==========================================
# 3. PSFViewerV6 主类
# ==========================================

class PSFViewerV6:
    def __init__(self, file_map, voxel_spacing, init_pos):
        self.file_map = file_map
        self.spacing = voxel_spacing
        self.cz, self.cx, self.cy = init_pos
        self.data_cache = {}
        self.names = list(file_map.keys())
        
        # 状态控制
        self.show_all_mode = False # False=Compare, True=ShowAll
        
        # 1. 寻找 GT (用于固定显示在第一行)
        self.gt_name = None
        if 'Ground Truth' in self.names:
            self.gt_name = 'Ground Truth'
        else:
            self.gt_name = self.names[0] # 没找到就默认第一个
        
        # 默认选中的实验模型 (排除掉 GT，选列表里的下一个)
        self.current_model = self.gt_name
        for n in self.names:
            if n != self.gt_name:
                self.current_model = n
                break
        
        # 2. 加载数据
        print(f"🚀 初始化 V6... GT is '{self.gt_name}'")
        for name, path in file_map.items():
            print(f"  -> 读取 {name}...")
            data = robust_load_nii(path)
            if data is not None:
                self.data_cache[name] = data
        
        if not self.data_cache: raise RuntimeError("无有效数据")
        
        # 3. 界面布局 (20x15 大屏)
        self.fig = plt.figure(figsize=(20, 15)) #, constrained_layout=True)
        
        # 主分割：左侧控制栏(15%)，右侧显示区(85%)
        gs_main = gridspec.GridSpec(1, 2, figure=self.fig, width_ratios=[0.15, 0.85], wspace=0.05)
        
        # --- 左侧控制区 ---
        gs_left = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs_main[0], height_ratios=[4, 1])
        
        # Radio Buttons
        self.ax_radio = self.fig.add_subplot(gs_left[0])
        self.ax_radio.set_title("Select Experiment", fontsize=16, fontweight='bold')
        self.ax_radio.set_facecolor('#f0f0f0')
        # 排除 GT，只显示可选的实验
        self.exp_names = [n for n in self.names if n != self.gt_name]
        # 如果没有实验模型，就放个空的
        if not self.exp_names: self.exp_names = [self.gt_name]
            
        self.radio = RadioButtons(self.ax_radio, self.exp_names, active=0)
        self.radio.on_clicked(self.change_model_view)
        
        # Mode Button
        self.ax_btn = self.fig.add_subplot(gs_left[1])
        self.btn = Button(self.ax_btn, 'Mode: Compare', color='lightblue', hovercolor='skyblue')
        self.btn.on_clicked(self.toggle_mode)
        
        # --- 右侧显示区 (三行) ---
        # 比例: GT(1) : Exp(1) : Profile(0.8)
        gs_right = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=gs_main[1], height_ratios=[1, 1, 0.8], hspace=0.25)
        
        # 第 1 行: GT Views (Fixed)
        gs_row1 = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=gs_right[0], wspace=0.05)
        self.ax_gt_ax  = self.fig.add_subplot(gs_row1[0])
        self.ax_gt_lat = self.fig.add_subplot(gs_row1[1])
        self.ax_gt_ele = self.fig.add_subplot(gs_row1[2])
        
        # 第 2 行: Exp Views (Dynamic)
        gs_row2 = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=gs_right[1], wspace=0.05)
        self.ax_exp_ax  = self.fig.add_subplot(gs_row2[0])
        self.ax_exp_lat = self.fig.add_subplot(gs_row2[1])
        self.ax_exp_ele = self.fig.add_subplot(gs_row2[2])
        
        # 第 3 行: Profiles
        gs_row3 = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=gs_right[2], wspace=0.15)
        self.ax_prof_z = self.fig.add_subplot(gs_row3[0])
        self.ax_prof_x = self.fig.add_subplot(gs_row3[1])
        self.ax_prof_y = self.fig.add_subplot(gs_row3[2])
        
        # 所有图像 Axes 列表，用于判断点击
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
        
        # 首次绘制
        self.update_all()
        print("\n✅ V6 双行对比版就绪！")
        plt.show()

    # ==========================================
    # 交互回调
    # ==========================================
    
    def change_model_view(self, label):
        self.current_model = label
        self.update_all()

    def toggle_mode(self, event):
        self.show_all_mode = not self.show_all_mode
        self.btn.label.set_text('Mode: Show All' if self.show_all_mode else 'Mode: Compare')
        self.plot_profiles() # 只需刷新曲线
        self.fig.canvas.draw_idle()

    def on_click(self, event):
        if event.inaxes not in self.slice_axes: return
        
        # 获取点击坐标
        col = int(event.xdata + 0.5)
        row = int(event.ydata + 0.5)
        
        # 判断点击的是哪个平面，更新 cx, cy, cz
        updated = False
        
        # Axial (XY) -> Clicked row=X, col=Y (注意 imshow 的 origin)
        if event.inaxes in [self.ax_gt_ax, self.ax_exp_ax]:
            if row < 128 and col < 128:
                self.cx, self.cy = row, col
                updated = True
                
        # Lateral (ZX) -> Clicked row=Z, col=X
        elif event.inaxes in [self.ax_gt_lat, self.ax_exp_lat]:
            if row < 1024 and col < 128:
                self.cz, self.cx = row, col
                updated = True
                
        # Elevation (ZY) -> Clicked row=Z, col=Y
        elif event.inaxes in [self.ax_gt_ele, self.ax_exp_ele]:
            if row < 1024 and col < 128:
                self.cz, self.cy = row, col
                updated = True
        
        if updated:
            print(f"🖱️  -> Jump to [Z={self.cz}, X={self.cx}, Y={self.cy}]")
            self.update_all()

    # ==========================================
    # 绘图逻辑
    # ==========================================
    
    def update_all(self):
        # 1. 画第一行 (GT)
        self.plot_slices_row(self.gt_name, 
                             [self.ax_gt_ax, self.ax_gt_lat, self.ax_gt_ele], 
                             title_prefix="[GT]")
        
        # 2. 画第二行 (Selected Exp)
        self.plot_slices_row(self.current_model, 
                             [self.ax_exp_ax, self.ax_exp_lat, self.ax_exp_ele], 
                             title_prefix="[Exp]")
        
        # 3. 画曲线
        self.plot_profiles()
        
        self.fig.canvas.draw_idle()

    def plot_slices_row(self, model_name, axes_list, title_prefix=""):
        """通用函数：画一行的三视图"""
        ax_ax, ax_lat, ax_ele = axes_list
        data = self.data_cache.get(model_name)
        
        if data is None: return

        # 十字光标样式
        cross_style = {'color': 'red', 'linestyle': '--', 'linewidth': 1.5, 'alpha': 0.8}
        blue_style = {'color': 'cyan', 'linestyle': '--', 'linewidth': 1.5, 'alpha': 0.8}
        
        # 1. Axial (Z=cz)
        img_axial = data[self.cz, :, :]
        ax_ax.clear()
        ax_ax.imshow(img_axial, cmap='gray', aspect='equal', vmin=-60, vmax=0)
        ax_ax.set_title(f"{title_prefix} Axial (Z={self.cz})\n{model_name}", fontsize=10, fontweight='bold')
        ax_ax.axvline(self.cy, **cross_style)
        ax_ax.axhline(self.cx, **cross_style)
        ax_ax.axis('off')

        # 2. Lateral (Y=cy)
        img_lat = data[:, :, self.cy]
        ax_lat.clear()
        ar_lat = self.spacing[0] / self.spacing[1]
        ax_lat.imshow(img_lat, cmap='gray', aspect=ar_lat, vmin=-60, vmax=0)
        ax_lat.set_title(f"{title_prefix} Lateral (Y={self.cy})", fontsize=10, fontweight='bold')
        ax_lat.axvline(self.cx, **cross_style)
        ax_lat.axhline(self.cz, **blue_style)
        ax_lat.axis('off')

        # 3. Elevation (X=cx)
        img_ele = data[:, self.cx, :]
        ax_ele.clear()
        ar_ele = self.spacing[0] / self.spacing[2]
        ax_ele.imshow(img_ele, cmap='gray', aspect=ar_ele, vmin=-60, vmax=0)
        ax_ele.set_title(f"{title_prefix} Elevation (X={self.cx})", fontsize=10, fontweight='bold')
        ax_ele.axvline(self.cy, **cross_style)
        ax_ele.axhline(self.cz, **blue_style)
        ax_ele.axis('off')

    def plot_profiles(self):
        """画底部的曲线图"""
        self.ax_prof_z.clear(); self.ax_prof_x.clear(); self.ax_prof_y.clear()
        metrics_buffer = []
        
        # 决定要画哪些模型
        if self.show_all_mode:
            # Show All: 画所有加载的模型
            target_models = self.names
        else:
            # Compare: 只画 GT 和 当前选中
            target_models = [self.gt_name, self.current_model]
            # 去重 (如果当前选中就是 GT)
            target_models = list(dict.fromkeys(target_models))
        
        # 绘图循环
        colors = ['k', 'r', 'g', 'b', 'm', 'c']
        
        for idx, name in enumerate(target_models):
            data = self.data_cache.get(name)
            if data is None: continue
            
            # 样式逻辑
            is_gt = (name == self.gt_name)
            is_selected = (name == self.current_model)
            
            if is_gt:
                color = 'k'     # GT 永远是黑色
                ls = '--'       # GT 永远是虚线
                lw = 2.5
                zorder = 10
            elif is_selected:
                color = 'r'     # 选中实验是红色
                ls = '-'        # 实线
                lw = 2.5
                zorder = 9
            else:
                # 其他实验 (Show All 模式下)
                color = colors[idx % len(colors)]
                ls = '-'
                lw = 1.0
                zorder = 1

            # 提取数据
            prof_z = data[:, self.cx, self.cy]
            prof_x = data[self.cz, :, self.cy]
            prof_y = data[self.cz, self.cx, :]
            
            # 计算
            fw_z, _, psl_z, ax_z, cv_z = compute_metrics(prof_z, self.spacing[0])
            fw_x, _, psl_x, ax_x, cv_x = compute_metrics(prof_x, self.spacing[1])
            fw_y, _, psl_y, ax_y, cv_y = compute_metrics(prof_y, self.spacing[2])
            
            # 绘制
            self.ax_prof_z.plot(ax_z - self.cz*self.spacing[0], cv_z, label=name, color=color, linestyle=ls, linewidth=lw, zorder=zorder)
            self.ax_prof_x.plot(ax_x - self.cx*self.spacing[1], cv_x, color=color, linestyle=ls, linewidth=lw, zorder=zorder)
            self.ax_prof_y.plot(ax_y - self.cy*self.spacing[2], cv_y, color=color, linestyle=ls, linewidth=lw, zorder=zorder)
            
            # 只有 GT 和 选中实验 才写入 CSV (避免 Show All 时写入太多垃圾数据)
            if is_gt or is_selected:
                metrics_buffer.append([self.cz, self.cx, self.cy, name, 
                                       f"{fw_z:.3f}", f"{fw_x:.3f}", f"{fw_y:.3f}",
                                       f"{psl_z:.1f}", f"{psl_x:.1f}", f"{psl_y:.1f}"])

        # 装饰 Axes
        for ax, title in zip([self.ax_prof_z, self.ax_prof_x, self.ax_prof_y], 
                             ['Axial Profile (Z)', 'Lateral Profile (X)', 'Elevation Profile (Y)']):
            ax.set_title(title, fontweight='bold')
            ax.set_ylim(-65, 5)
            ax.grid(True, linestyle='--', alpha=0.5)
            ax.axhline(-6, color='gray', linestyle=':', linewidth=1.5)

        # 图例
        self.ax_prof_z.legend(fontsize=10, loc='lower center', framealpha=0.9)
        
        # 写入 CSV
        if metrics_buffer:
            with open(OUTPUT_CSV, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerows(metrics_buffer)

if __name__ == '__main__':
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    viewer = PSFViewerV6(file_dict, VOXEL_SPACING, INITIAL_POS)