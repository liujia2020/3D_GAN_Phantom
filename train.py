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
    
    # 3D -> 2D 切片 (取中间层)
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

# =========================================================================
# 辅助函数：计算 PSNR
# =========================================================================
def calculate_psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * np.log10(PIXEL_MAX / np.sqrt(mse))

# =========================================================================
# 主训练循环
# =========================================================================
if __name__ == '__main__':
    opt = TrainOptions().parse()
    dataset = create_dataset(opt)
    dataset_size = len(dataset)
    print(f'The number of training images = {dataset_size}')

    model = create_model(opt)
    model.setup(opt) # 这里会自动打印网络结构

    total_iters = 0
    
    # [CSV] 初始化日志文件
    expr_dir = os.path.join(opt.checkpoints_dir, opt.name)
    if not os.path.exists(expr_dir):
        os.makedirs(expr_dir)
    log_name = os.path.join(expr_dir, 'loss_log.csv')
    
    # 获取 loss 名字用于表头
    loss_names = model.loss_names
    
    if not os.path.exists(log_name):
        with open(log_name, mode='w', newline='') as f:
            header = ['epoch', 'time'] + loss_names + ['psnr_train', 'lr']
            csv.writer(f).writerow(header)

    print(">>> Start Training Loop...")

    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):
        epoch_start_time = time.time()
        iter_data_time = time.time()
        epoch_iter = 0
        
        # 用于计算 Epoch 平均 Loss 和 PSNR
        epoch_loss_sum = {name: 0.0 for name in loss_names}
        epoch_psnr_sum = 0.0
        num_batch = 0

        # Tqdm 进度条
        with tqdm(total=len(dataset), desc=f"Epoch {epoch}/{opt.n_epochs + opt.n_epochs_decay}", unit="img") as pbar:
            for i, data in enumerate(dataset):
                iter_start_time = time.time()
                if total_iters % opt.print_freq == 0:
                    t_data = iter_start_time - iter_data_time

                total_iters += opt.batch_size
                epoch_iter += opt.batch_size
                
                # 1. 训练核心步
                model.set_input(data)
                model.optimize_parameters()

                # 2. 获取 Loss
                losses = model.get_current_losses()
                for name in losses:
                    epoch_loss_sum[name] += losses[name]
                
                # 3. 获取 Visuals (用于计算训练集 PSNR 监控)
                # [关键修正] 适配 AuganModel 的新命名
                visuals = model.get_current_visuals()
                
                # 安全提取: 使用 .get() 防止键名不存在报错
                # 映射: input_lq -> input, fake_sq -> fake, real_sq -> real
                input_tensor = visuals.get('input_lq')
                fake_tensor  = visuals.get('fake_sq')
                real_tensor  = visuals.get('real_sq')
                
                # 如果找不到新名字，尝试回退到旧名字 (兼容性保护)
                if input_tensor is None: input_tensor = visuals.get('real_A')
                if fake_tensor is None:  fake_tensor  = visuals.get('fake_B')
                if real_tensor is None:  real_tensor  = visuals.get('real_B')

                # 计算当前 Batch 的 PSNR
                current_psnr = 0.0
                if fake_tensor is not None and real_tensor is not None:
                    # 简单转 numpy 计算，不绘图
                    # 注意：这里为了速度，直接取 tensor 数据计算，可能需要简化的 tensor2im
                    # 为了不拖慢训练，我们只在 print_freq 时计算或者只累加
                    # 这里我们简单估算：
                    img_f = tensor2im_custom(fake_tensor)
                    img_r = tensor2im_custom(real_tensor)
                    current_psnr = calculate_psnr(img_f, img_r)
                    epoch_psnr_sum += current_psnr

                num_batch += 1

                # 更新进度条
                pbar.set_postfix(**losses, psnr=f"{current_psnr:.2f}")
                pbar.update(opt.batch_size)
                
                iter_data_time = time.time()

        # End of Epoch
        # 更新学习率
        model.update_learning_rate()
        
        # 计算平均统计
        for name in epoch_loss_sum:
            epoch_loss_sum[name] /= num_batch
        avg_psnr = epoch_psnr_sum / num_batch
        time_taken = time.time() - epoch_start_time
        
        # 获取当前 LR
        current_lr = model.optimizers[0].param_groups[0]['lr']

        # 1. 打印日志
        loss_str = " | ".join([f"{k}: {v:.4f}" for k, v in epoch_loss_sum.items()])
        log_msg = (
            f"\n{'-'*60}\n"
            f"  ✅ [End of Epoch {epoch}] \n"
            f"  📉 [Loss Avg]      |  {loss_str}\n"
            f"  📊 [Metrics Avg]   |  PSNR: {avg_psnr:.2f} dB  |  Time: {time_taken:.1f}s  |  LR: {current_lr:.6f}\n"
            f"{'-'*60}\n"
        )
        print(log_msg)

        # 2. 写入 CSV
        with open(log_name, mode='a', newline='') as f:
            row = [epoch, f"{time_taken:.1f}"]
            for name in loss_names:
                # 使用 .get 此时更安全
                row.append(f"{epoch_loss_sum.get(name, 0.0):.4f}")
            row.append(f"{avg_psnr:.2f}")
            row.append(f"{current_lr:.6f}")
            csv.writer(f).writerow(row)

        # 3. [核心修改] 生成 Paper Style 对比图 (每个 Epoch 保存一张)
        if input_tensor is not None and fake_tensor is not None and real_tensor is not None:
            img_dir = os.path.join(expr_dir, 'web_images')
            if not os.path.exists(img_dir): os.makedirs(img_dir)
            
            # 转换当前这张图的 Numpy 数据
            img_lq = tensor2im_custom(input_tensor)
            img_fake = tensor2im_custom(fake_tensor)
            img_sq = tensor2im_custom(real_tensor)
            
            # 计算这张展示图片的具体 PSNR (所见即所得)
            slice_psnr = calculate_psnr(img_fake, img_sq)
            
            # 使用 Matplotlib 绘图
            fig, axes = plt.subplots(1, 3, figsize=(12, 4))
            
            # (A) Input
            axes[0].imshow(img_lq.astype(np.uint8))
            axes[0].set_title("Input (Low Quality)")
            axes[0].axis('off')
            
            # (B) AUGAN Output
            axes[1].imshow(img_fake.astype(np.uint8))
            axes[1].set_title(f"AUGAN (PSNR: {slice_psnr:.2f} dB)")
            axes[1].axis('off')
            
            # (C) Ground Truth
            axes[2].imshow(img_sq.astype(np.uint8))
            axes[2].set_title("Ground Truth")
            axes[2].axis('off')
            
            plt.tight_layout()
            save_path = os.path.join(img_dir, f'epoch_{epoch}_train_preview.png')
            plt.savefig(save_path, dpi=150)
            plt.close()
            print(f"  📸 Saved preview to: {save_path}")

        # 保存模型 (每5个epoch或自定义)
        if epoch % opt.save_epoch_freq == 0:
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            model.save_networks('latest')
            model.save_networks(epoch)
        
        # 强制保存 latest
        model.save_networks('latest')

    print(">>> Training Finished!")