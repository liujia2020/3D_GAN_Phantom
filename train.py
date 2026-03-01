import time
import os
import torch
import numpy as np
import csv
import sys
import nibabel as nib  # [新增] 用于读取固定的 NIfTI 监控文件
from tqdm import tqdm
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model

# === 引入 Matplotlib ===
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# =========================================================================
# 辅助函数：Tensor转图 (自动处理 3D 切片)
# =========================================================================
def tensor2im_custom(input_image, imtype=np.uint8):
    if not isinstance(input_image, torch.Tensor):
        return input_image
    
    image_tensor = input_image.data
    image_numpy = image_tensor[0].cpu().float().numpy() 
    
    if image_numpy.ndim == 4:
        mid_slice = image_numpy.shape[1] // 2
        image_numpy = image_numpy[:, mid_slice, :, :]
        
    if image_numpy.shape[0] == 1:
        image_numpy = np.tile(image_numpy, (3, 1, 1))
        
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    image_numpy = np.clip(image_numpy, 0, 255).astype(imtype)
    
    return image_numpy

def calculate_psnr(img1, img2):
    mse = np.mean((img1.astype(np.float32) - img2.astype(np.float32)) ** 2)
    if mse == 0: return 100
    return 20 * np.log10(255.0 / np.sqrt(mse))

# =========================================================================
# [核心功能] 绘制论文级监控图
# =========================================================================
def save_paper_style_fig(save_path, epoch, exp_name, psnr_val, img_lq, img_fake, img_sq):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6.5))
    
    header_txt = f"Exp: {exp_name}  |  Epoch: {epoch}  |  Fixed Sample PSNR: {psnr_val:.2f} dB"
    fig.suptitle(header_txt, fontsize=20, fontweight='bold', y=0.92)
    
    items = [
        ('Input (Low Quality)', img_lq),
        ('Generated (High Quality)', img_fake),
        ('Ground Truth (High Quality)', img_sq)
    ]
    
    for ax, (title, img) in zip(axes, items):
        ax.imshow(img)
        ax.set_title(title, fontsize=16, pad=10, fontweight='medium')
        ax.axis('off') 
    
    plt.subplots_adjust(top=0.85, wspace=0.05, left=0.02, right=0.98, bottom=0.02)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

# =========================================================================
# 主训练逻辑
# =========================================================================
if __name__ == '__main__':
    opt = TrainOptions().parse()
    
    print("\n" + "="*80)
    print(f"🚀 训练启动 | 实验: {opt.name}")
    print(f"📂 数据路径: {opt.dataroot}")
    print("="*80 + "\n")

    dataset = create_dataset(opt)
    dataset_size = len(dataset) * opt.batch_size 
    
    model = create_model(opt)
    model.setup(opt)
    
    # =========================================================================
    # [核心修改 1]：在训练开始前，预先加载我们做好的“定点监控样本”
    # =========================================================================
    monitor_lq_path = './monitor_data/monitor_LQ.nii'
    monitor_hq_path = './monitor_data/monitor_HQ.nii'
    
    has_monitor = os.path.exists(monitor_lq_path) and os.path.exists(monitor_hq_path)
    mon_lq_tensor, mon_hq_tensor = None, None
    

    if has_monitor:
        print("🔍 发现固定监控样本，正在加载...")
        # 读取 NIfTI 数据并转为 float32
        mon_lq_np = nib.load(monitor_lq_path).get_fdata().astype(np.float32)
        mon_hq_np = nib.load(monitor_hq_path).get_fdata().astype(np.float32)
        
        # ==========================================================
        # [归一化逻辑]：映射到 [-1, 1]
        # ==========================================================
        norm_min = getattr(opt, 'norm_min', -60.0)
        norm_max = getattr(opt, 'norm_max', 0.0)
        
        mon_lq_np = (mon_lq_np - norm_min) / (norm_max - norm_min)
        mon_hq_np = (mon_hq_np - norm_min) / (norm_max - norm_min)
        mon_lq_np = mon_lq_np * 2.0 - 1.0
        mon_hq_np = mon_hq_np * 2.0 - 1.0
        mon_lq_np = np.clip(mon_lq_np, -1.0, 1.0)
        mon_hq_np = np.clip(mon_hq_np, -1.0, 1.0)
        
        # ==========================================================
        # [核心修复：2.5D 降维改造]：提取中心层，抛弃 3D 体积
        # ==========================================================
        # 假设 mon_lq_np 形状是 (D, H, W)，比如 (128, 64, 64)
        d, h, w = mon_lq_np.shape
        z = d // 2  # 取最中间的一层作为固定监控切片
        z_prev = max(0, z - 1)
        z_next = min(d - 1, z + 1)
        
        # 为 LQ 切出 3 层
        lq_slice_prev = mon_lq_np[z_prev, :, :]
        lq_slice_curr = mon_lq_np[z, :, :]
        lq_slice_next = mon_lq_np[z_next, :, :]
        
        # 叠成 (3, H, W) 和 (1, H, W)
        mon_lq_25d = np.stack([lq_slice_prev, lq_slice_curr, lq_slice_next], axis=0)
        mon_hq_2d = np.expand_dims(mon_hq_np[z, :, :], axis=0)
        
        # 升维加上 Batch 维度: (1, Channel, H, W)
        mon_lq_tensor = torch.from_numpy(mon_lq_25d).unsqueeze(0).to(model.device)
        mon_hq_tensor = torch.from_numpy(mon_hq_2d).unsqueeze(0).to(model.device)
        
        print(f"✅ 固定监控样本已转换为 2.5D！LQ={mon_lq_tensor.shape}, HQ={mon_hq_tensor.shape}")
    else:
        print("⚠️ 未检测到 monitor_data，请确认你是否运行过 crop_monitor_patch.py！")
        
    expr_dir = os.path.join(opt.checkpoints_dir, opt.name)
    if not os.path.exists(expr_dir): os.makedirs(expr_dir)
    log_name = os.path.join(expr_dir, 'loss_log.csv')
    loss_names = model.loss_names 
    
    if not opt.continue_train or not os.path.exists(log_name):
        with open(log_name, mode='w', newline='') as f:
            header = ['Epoch', 'Time(s)'] + loss_names + ['Train_PSNR', 'LR']
            csv.writer(f).writerow(header)

    total_iters = 0                 
    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):
        epoch_start_time = time.time()
        
        epoch_loss_sum = {name: 0.0 for name in loss_names}
        epoch_psnr_sum = 0.0
        num_batches = 0

        pbar = tqdm(enumerate(dataset), total=len(dataset), desc=f"Epoch {epoch}", file=sys.stdout)

        for i, data in pbar:
            iter_start_time = time.time()
            total_iters += opt.batch_size
            
            model.set_input(data)
            model.optimize_parameters()
            
            errors = model.get_current_losses()
            for k, v in errors.items():
                if k in epoch_loss_sum: epoch_loss_sum[k] += v

            # 计算训练集的平均 PSNR 用于进度条显示
            fake_tensor = getattr(model, 'fake_sq', None)
            real_tensor = getattr(model, 'real_sq', None)
            
            current_psnr = 0.0
            if fake_tensor is not None and real_tensor is not None:
                fake_im_raw = tensor2im_custom(fake_tensor)
                real_im_raw = tensor2im_custom(real_tensor)
                current_psnr = calculate_psnr(fake_im_raw, real_im_raw)
                epoch_psnr_sum += current_psnr

            num_batches += 1

            pbar.set_postfix({
                'L1': f"{errors.get('G_L1', errors.get('G_Pixel', 0)):.3f}", 
                'PSNR': f"{current_psnr:.1f}"
            })

            if total_iters % opt.save_latest_freq == 0:
                model.save_networks('latest')

        # === Epoch 结束结算 ===
        if num_batches > 0:
            for k in epoch_loss_sum: epoch_loss_sum[k] /= num_batches
            avg_psnr = epoch_psnr_sum / num_batches
        else:
            avg_psnr = 0.0
        
        time_taken = time.time() - epoch_start_time
        model.update_learning_rate()
        current_lr = model.optimizers[0].param_groups[0]['lr']

        # 1. 打印仪表盘 Log
        gen_losses = []
        disc_losses = []
        for k, v in epoch_loss_sum.items():
            if k.startswith('G_'):
                gen_losses.append(f"{k.replace('G_', '')}: {v:.4f}")
            elif k.startswith('D_'):
                disc_losses.append(f"{k.replace('D_', '')}: {v:.4f}")

        log_msg = (
            f"\n{'='*20} Epoch {epoch} Summary {'='*20}\n"
            f"  🎨 [Generator Avg] |  {'  |  '.join(gen_losses)}\n"
            f"  ⚖️  [Discriminator] |  {'  |  '.join(disc_losses)}\n"
            f"  📊 [Metrics Avg]   |  Train PSNR: {avg_psnr:.2f} dB  |  Time: {time_taken:.1f}s  |  LR: {current_lr:.6f}\n"
            f"{'-'*60}\n"
        )
        print(log_msg)

        # 2. 写入 CSV
        with open(log_name, mode='a', newline='') as f:
            row = [epoch, f"{time_taken:.1f}"]
            for name in loss_names:
                row.append(f"{epoch_loss_sum.get(name, 0.0):.4f}")
            row.append(f"{avg_psnr:.2f}")
            row.append(f"{current_lr:.6f}")
            csv.writer(f).writerow(row)

        # =========================================================================
        # [核心修改 2]：在每个 Epoch 结束时，将固定样本喂给模型，并保存监控图
        # =========================================================================
        if has_monitor:
            img_dir = os.path.join(expr_dir, 'web_images')
            if not os.path.exists(img_dir): os.makedirs(img_dir)
            
            # 开启 eval 模式，关闭梯度，这能保证这次推理绝对不污染训练进度
            model.netG.eval()
            with torch.no_grad():
                mon_fake_tensor = model.netG(mon_lq_tensor)
            model.netG.train() # 推理完立刻切回训练模式！
            
            # 转为 numpy 图片格式
            img_lq = tensor2im_custom(mon_lq_tensor)
            img_fake = tensor2im_custom(mon_fake_tensor)
            img_sq = tensor2im_custom(mon_hq_tensor)
            
            # 计算专门针对这张神仙图的 PSNR
            slice_psnr = calculate_psnr(img_fake, img_sq)
            
            # 保存雷打不动的定点图
            save_path = os.path.join(img_dir, f'epoch_{epoch:03d}_comparison.png')
            save_paper_style_fig(save_path, epoch, opt.name, slice_psnr, img_lq, img_fake, img_sq)

        # 4. 保存模型
        if epoch % opt.save_epoch_freq == 0:
            model.save_networks('latest')
            model.save_networks(epoch)

    print("🏁 所有训练完成!")

# import time
# import os
# import torch
# import numpy as np
# import csv
# import sys
# from tqdm import tqdm
# from options.train_options import TrainOptions
# from data import create_dataset
# from models import create_model

# # === [引入 Matplotlib] ===
# import matplotlib
# matplotlib.use('Agg') # 确保在无屏幕服务器上也能运行
# import matplotlib.pyplot as plt

# # =========================================================================
# # 辅助函数：Tensor转图 (自动处理 3D 切片)
# # =========================================================================
# def tensor2im_custom(input_image, imtype=np.uint8):
#     """
#     将 Tensor 转换为 Numpy 图像，自动处理 3D 数据的中间切片
#     """
#     if not isinstance(input_image, torch.Tensor):
#         return input_image
    
#     image_tensor = input_image.data
#     image_numpy = image_tensor[0].cpu().float().numpy() # 取 Batch 第一个 -> (C, D, H, W) 或 (C, H, W)
    
#     # [核心适配] 3D -> 2D 切片
#     # 如果维度是 4 (C, D, H, W)，说明是 3D 数据
#     if image_numpy.ndim == 4:
#         mid_slice = image_numpy.shape[1] // 2
#         image_numpy = image_numpy[:, mid_slice, :, :] # -> (C, H, W)
        
#     # 单通道 -> RGB (复制通道)
#     if image_numpy.shape[0] == 1:
#         image_numpy = np.tile(image_numpy, (3, 1, 1))
        
#     # 反归一化 (-1, 1) -> (0, 255)
#     image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
#     image_numpy = np.clip(image_numpy, 0, 255).astype(imtype)
    
#     return image_numpy

# def calculate_psnr(img1, img2):
#     mse = np.mean((img1.astype(np.float32) - img2.astype(np.float32)) ** 2)
#     if mse == 0: return 100
#     return 20 * np.log10(255.0 / np.sqrt(mse))

# # =========================================================================
# # [核心功能] 绘制论文级监控图
# # =========================================================================
# def save_paper_style_fig(save_path, epoch, exp_name, psnr_val, img_lq, img_fake, img_sq):
#     """
#     绘制 Input / Generated / Truth 对比图
#     """
#     fig, axes = plt.subplots(1, 3, figsize=(18, 6.5))
    
#     header_txt = f"Exp: {exp_name}  |  Epoch: {epoch}  |  Slice PSNR: {psnr_val:.2f} dB"
#     fig.suptitle(header_txt, fontsize=20, fontweight='bold', y=0.92)
    
#     items = [
#         ('Input (Low Quality)', img_lq),
#         ('Generated (High Quality)', img_fake),
#         ('Ground Truth (High Quality)', img_sq)
#     ]
    
#     for ax, (title, img) in zip(axes, items):
#         ax.imshow(img)
#         ax.set_title(title, fontsize=16, pad=10, fontweight='medium')
#         ax.axis('off') 
    
#     plt.subplots_adjust(top=0.85, wspace=0.05, left=0.02, right=0.98, bottom=0.02)
#     plt.savefig(save_path, dpi=150, bbox_inches='tight')
#     plt.close(fig)

# # =========================================================================
# # 主训练逻辑
# # =========================================================================
# if __name__ == '__main__':
#     opt = TrainOptions().parse()
    
#     print("\n" + "="*80)
#     print(f"🚀 训练启动 | 实验: {opt.name}")
#     print(f"📂 数据路径: {opt.dataroot}")
#     print("="*80 + "\n")

#     dataset = create_dataset(opt)
#     dataset_size = len(dataset) * opt.batch_size 
    
#     model = create_model(opt)
#     model.setup(opt)
    
#     expr_dir = os.path.join(opt.checkpoints_dir, opt.name)
#     if not os.path.exists(expr_dir): os.makedirs(expr_dir)
#     log_name = os.path.join(expr_dir, 'loss_log.csv')
#     loss_names = model.loss_names 
    
#     if not opt.continue_train or not os.path.exists(log_name):
#         with open(log_name, mode='w', newline='') as f:
#             header = ['Epoch', 'Time(s)'] + loss_names + ['PSNR', 'LR']
#             csv.writer(f).writerow(header)

#     total_iters = 0                 
#     for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):
#         epoch_start_time = time.time()
        
#         epoch_loss_sum = {name: 0.0 for name in loss_names}
#         epoch_psnr_sum = 0.0
#         num_batches = 0

#         # 进度条
#         pbar = tqdm(enumerate(dataset), total=len(dataset), desc=f"Epoch {epoch}", file=sys.stdout)

#         for i, data in pbar:
#             iter_start_time = time.time()
#             total_iters += opt.batch_size
            
#             model.set_input(data)
#             model.optimize_parameters()
            
#             # Loss 记录
#             errors = model.get_current_losses()
#             for k, v in errors.items():
#                 if k in epoch_loss_sum: epoch_loss_sum[k] += v

#             # [核心修复] 提取 Tensor 用于计算和显示 (适配新版 AuganModel 变量名)
#             # 优先获取 fake_sq (5D), real_sq (5D), input_lq (5D)
#             fake_tensor = getattr(model, 'fake_sq', None)
#             real_tensor = getattr(model, 'real_sq', None)
#             input_tensor = getattr(model, 'input_lq', None)
            
#             # 如果找不到 (比如测试模式或者变量名变了)，尝试回退到旧名称 (虽然现在应该都统一了)
#             if fake_tensor is None: fake_tensor = getattr(model, 'fake_B', None)
#             if real_tensor is None: real_tensor = getattr(model, 'real_B', None)
#             if input_tensor is None: input_tensor = getattr(model, 'input_A', None)
            
#             # 计算 PSNR (Log用)
#             current_psnr = 0.0
#             if fake_tensor is not None and real_tensor is not None:
#                 fake_im_raw = tensor2im_custom(fake_tensor)
#                 real_im_raw = tensor2im_custom(real_tensor)
#                 current_psnr = calculate_psnr(fake_im_raw, real_im_raw)
#                 epoch_psnr_sum += current_psnr

#             num_batches += 1

#             # 进度条显示
#             pbar.set_postfix({
#                 'L1': f"{errors.get('G_L1', errors.get('G_Pixel', 0)):.3f}", # 兼容 G_L1 和 G_Pixel
#                 'PSNR': f"{current_psnr:.1f}"
#             })

#             # 定期保存模型
#             if total_iters % opt.save_latest_freq == 0:
#                 model.save_networks('latest')

#         # === Epoch 结束结算 ===
#         if num_batches > 0:
#             for k in epoch_loss_sum: epoch_loss_sum[k] /= num_batches
#             avg_psnr = epoch_psnr_sum / num_batches
#         else:
#             avg_psnr = 0.0
        
#         time_taken = time.time() - epoch_start_time
#         model.update_learning_rate()
#         current_lr = model.optimizers[0].param_groups[0]['lr']

#         # 1. 打印仪表盘 Log
#         gen_losses = []
#         disc_losses = []
#         for k, v in epoch_loss_sum.items():
#             if k.startswith('G_'):
#                 gen_losses.append(f"{k.replace('G_', '')}: {v:.4f}")
#             elif k.startswith('D_'):
#                 disc_losses.append(f"{k.replace('D_', '')}: {v:.4f}")

#         log_msg = (
#             f"\n{'='*20} Epoch {epoch} Summary {'='*20}\n"
#             f"  🎨 [Generator Avg] |  {'  |  '.join(gen_losses)}\n"
#             f"  ⚖️  [Discriminator] |  {'  |  '.join(disc_losses)}\n"
#             f"  📊 [Metrics Avg]   |  PSNR: {avg_psnr:.2f} dB  |  Time: {time_taken:.1f}s  |  LR: {current_lr:.6f}\n"
#             f"{'-'*60}\n"
#         )
#         print(log_msg)

#         # 2. 写入 CSV
#         with open(log_name, mode='a', newline='') as f:
#             row = [epoch, f"{time_taken:.1f}"]
#             for name in loss_names:
#                 row.append(f"{epoch_loss_sum.get(name, 0.0):.4f}") # 使用 .get 防止 Key 缺失报错
#             row.append(f"{avg_psnr:.2f}")
#             row.append(f"{current_lr:.6f}")
#             csv.writer(f).writerow(row)

#         # 3. 生成 Paper Style 对比图
#         if input_tensor is not None and fake_tensor is not None and real_tensor is not None:
#             img_dir = os.path.join(expr_dir, 'web_images')
#             if not os.path.exists(img_dir): os.makedirs(img_dir)
            
#             # 转换当前这张图的 Numpy 数据
#             img_lq = tensor2im_custom(input_tensor)
#             img_fake = tensor2im_custom(fake_tensor)
#             img_sq = tensor2im_custom(real_tensor)
            
#             # 计算这张展示图片的具体 PSNR
#             slice_psnr = calculate_psnr(img_fake, img_sq)
            
#             # 绘制大图
#             save_path = os.path.join(img_dir, f'epoch_{epoch:03d}_comparison.png')
#             save_paper_style_fig(save_path, epoch, opt.name, slice_psnr, img_lq, img_fake, img_sq)

#         # 4. 保存模型
#         if epoch % opt.save_epoch_freq == 0:
#             model.save_networks('latest')
#             model.save_networks(epoch)

#     print("🏁 所有训练完成!")