import time
import os
import torch
import numpy as np
import csv
import sys
from tqdm import tqdm
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model

# === [新增] 引入 Matplotlib 用于专业绘图 ===
import matplotlib
matplotlib.use('Agg') # 确保在无屏幕服务器上也能运行
import matplotlib.pyplot as plt

# =========================================================================
# 辅助函数：Tensor转图 (保持 scale_factor=1 用于 Matplotlib 输入)
# =========================================================================
def tensor2im_custom(input_image, imtype=np.uint8):
    if isinstance(input_image, torch.Tensor):
        image_tensor = input_image.data
    else:
        return input_image
    
    image_numpy = image_tensor[0].cpu().float().numpy()
    
    # 3D -> 2D 切片
    if image_numpy.ndim == 4:
        mid_slice = image_numpy.shape[1] // 2
        image_numpy = image_numpy[:, mid_slice, :, :]
        
    # 单通道 -> RGB
    if image_numpy.shape[0] == 1:
        image_numpy = np.tile(image_numpy, (3, 1, 1))
        
    # 反归一化
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
    """
    使用 Matplotlib 绘制布局精美的对比图
    结构：
    [          Header Info (Epoch, Exp, PSNR)           ]
    [ Input Title ]  [ Generated Title ]  [ Truth Title ]
    [   Image 1   ]  [     Image 2     ]  [   Image 3   ]
    """
    # 1. 设置画布 (宽18英寸, 高6英寸 -> 高清大图)
    fig, axes = plt.subplots(1, 3, figsize=(18, 6.5))
    
    # 2. 构造顶部大标题
    header_txt = f"Exp: {exp_name}  |  Epoch: {epoch}  |  Slice PSNR: {psnr_val:.2f} dB"
    fig.suptitle(header_txt, fontsize=20, fontweight='bold', y=0.92)
    
    # 3. 准备数据
    items = [
        ('Input (Low Quality)', img_lq),
        ('Generated (High Quality)', img_fake),
        ('Ground Truth (High Quality)', img_sq)
    ]
    
    # 4. 循环绘制子图
    for ax, (title, img) in zip(axes, items):
        ax.imshow(img)
        ax.set_title(title, fontsize=16, pad=10, fontweight='medium')
        ax.axis('off') # 去掉难看的坐标轴刻度
    
    # 5. 调整间距并保存
    plt.subplots_adjust(top=0.85, wspace=0.05, left=0.02, right=0.98, bottom=0.02)
    
    # dpi=150 保证文字和图像都非常清晰
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig) # 关闭画布释放内存

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
    
    expr_dir = os.path.join(opt.checkpoints_dir, opt.name)
    if not os.path.exists(expr_dir): os.makedirs(expr_dir)
    log_name = os.path.join(expr_dir, 'loss_log.csv')
    loss_names = model.loss_names 
    
    if not opt.continue_train or not os.path.exists(log_name):
        with open(log_name, mode='w', newline='') as f:
            header = ['Epoch', 'Time(s)'] + loss_names + ['PSNR', 'LR']
            csv.writer(f).writerow(header)

    total_iters = 0                
    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):
        epoch_start_time = time.time()
        
        epoch_loss_sum = {name: 0.0 for name in loss_names}
        epoch_psnr_sum = 0.0
        num_batches = 0

        # 进度条
        pbar = tqdm(enumerate(dataset), total=len(dataset), desc=f"Epoch {epoch}", file=sys.stdout)

        for i, data in pbar:
            iter_start_time = time.time()
            total_iters += opt.batch_size
            
            model.set_input(data)
            model.optimize_parameters()
            
            # Loss 记录
            errors = model.get_current_losses()
            for k, v in errors.items():
                if k in epoch_loss_sum: epoch_loss_sum[k] += v

            # 提取 Tensor 用于计算和显示
            fake_tensor = getattr(model, 'fake_hq', getattr(model, 'fake_B', None))
            real_tensor = getattr(model, 'real_sq', getattr(model, 'real_B', None))
            input_tensor = getattr(model, 'real_lq', getattr(model, 'real_A', None))
            
            # 计算 PSNR (Log用)
            current_psnr = 0.0
            if fake_tensor is not None and real_tensor is not None:
                # 这里的 scale_factor=1 保证计算准确
                fake_im_raw = tensor2im_custom(fake_tensor)
                real_im_raw = tensor2im_custom(real_tensor)
                current_psnr = calculate_psnr(fake_im_raw, real_im_raw)
                epoch_psnr_sum += current_psnr

            num_batches += 1

            # 进度条小尾巴
            pbar.set_postfix({
                'L1': f"{errors.get('G_Pixel', 0):.3f}",
                'PSNR': f"{current_psnr:.1f}"
            })

            # 定期保存模型
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
            f"  📊 [Metrics Avg]   |  PSNR: {avg_psnr:.2f} dB  |  Time: {time_taken:.1f}s  |  LR: {current_lr:.6f}\n"
            f"{'-'*60}\n"
        )
        print(log_msg)

        # 2. 写入 CSV
        with open(log_name, mode='a', newline='') as f:
            row = [epoch, f"{time_taken:.1f}"]
            for name in loss_names:
                row.append(f"{epoch_loss_sum[name]:.4f}")
            row.append(f"{avg_psnr:.2f}")
            row.append(f"{current_lr:.6f}")
            csv.writer(f).writerow(row)

        # 3. [核心修改] 生成 Paper Style 对比图
        if input_tensor is not None and fake_tensor is not None and real_tensor is not None:
            img_dir = os.path.join(expr_dir, 'web_images')
            if not os.path.exists(img_dir): os.makedirs(img_dir)
            
            # 转换当前这张图的 Numpy 数据
            img_lq = tensor2im_custom(input_tensor)
            img_fake = tensor2im_custom(fake_tensor)
            img_sq = tensor2im_custom(real_tensor)
            
            # 计算这张展示图片的具体 PSNR (所见即所得)
            slice_psnr = calculate_psnr(img_fake, img_sq)
            
            # 绘制大图
            save_path = os.path.join(img_dir, f'epoch_{epoch:03d}_comparison.png')
            save_paper_style_fig(save_path, epoch, opt.name, slice_psnr, img_lq, img_fake, img_sq)

        # 4. 保存模型
        if epoch % opt.save_epoch_freq == 0:
            model.save_networks('latest')
            model.save_networks(epoch)

    print("🏁 所有训练完成!")