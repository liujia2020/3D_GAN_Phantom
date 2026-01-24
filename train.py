import time
import os
import torch
import numpy as np
import csv
from PIL import Image
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
from util.visualizer import Visualizer

# =========================================================================
# 辅助函数：将 Tensor 转为可视化图片 (单通道 -> 灰度图)
# =========================================================================
def tensor2im_custom(input_image, imtype=np.uint8):
    """
    将 [-1, 1] 的 Tensor 转换为 [0, 255] 的 numpy image
    """
    if isinstance(input_image, torch.Tensor):
        image_tensor = input_image.data
    else:
        return input_image
    
    image_numpy = image_tensor[0].cpu().float().numpy()  # 取 Batch 中的第一张
    
    # 形状处理: (C, H, W) -> (H, W, C)
    if image_numpy.shape[0] == 1:  # 单通道 (灰度)
        image_numpy = np.tile(image_numpy, (3, 1, 1))  # 复制成 3 通道方便显示
    
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0  # [-1,1] -> [0,255]
    return image_numpy.astype(imtype)

# =========================================================================
# 辅助函数：计算 PSNR (用于监控训练质量)
# =========================================================================
def calculate_psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * np.log10(PIXEL_MAX / np.sqrt(mse))

# =========================================================================
# 主训练逻辑
# =========================================================================
if __name__ == '__main__':
    # 1. 解析参数
    opt = TrainOptions().parse()
    
    # ------------------------------------------------
    # [增强功能 1] 详细信息打印与固化
    # ------------------------------------------------
    print("="*80)
    print(f"🚀 训练启动: {opt.name}")
    print(f"📂 数据集路径: {opt.dataroot}")
    print(f"   输入文件夹 (LQ): {opt.dir_lq}")
    print(f"   真值文件夹 (SQ): {opt.dir_sq}")
    print(f"🔧 核心参数: Batch={opt.batch_size}, L1_W={opt.lambda_pixel}, GAN_W={opt.lambda_gan}, VGG_W={opt.lambda_perceptual}")
    print("="*80)

    # 2. 创建数据集
    dataset = create_dataset(opt)
    dataset_size = len(dataset)
    print(f'📊 训练集图片总数 = {dataset_size}')

    # 3. 创建模型
    model = create_model(opt)
    model.setup(opt)
    
    # 4. 初始化 CSV 日志文件
    # log_path: ./checkpoints/实验名/loss_log.csv
    expr_dir = os.path.join(opt.checkpoints_dir, opt.name)
    if not os.path.exists(expr_dir):
        os.makedirs(expr_dir)
    log_name = os.path.join(expr_dir, 'loss_log.csv')
    
    # 如果是新训练，写入表头；如果是续训，直接追加
    if not opt.continue_train or not os.path.exists(log_name):
        with open(log_name, mode='w', newline='') as f:
            writer = csv.writer(f)
            # 表头：Epoch, 耗时, G总Loss, D总Loss, L1项, GAN项, VGG项, PSNR, 学习率
            writer.writerow(['Epoch', 'Time(s)', 'G_Total', 'D_Total', 'G_L1', 'G_GAN', 'G_VGG', 'PSNR', 'LR'])
        print(f"📝 创建新日志文件: {log_name}")
    else:
        print(f"🔄 追加到现有日志文件: {log_name}")

    # 5. 训练循环
    total_iters = 0                
    
    # 确定起止 Epoch
    start_epoch = opt.epoch_count
    end_epoch = opt.n_epochs + opt.n_epochs_decay

    for epoch in range(start_epoch, end_epoch + 1):
        epoch_start_time = time.time()
        iter_data_time = time.time()
        epoch_iter = 0
        model.update_learning_rate()

        # 用于统计本 Epoch 的平均 Loss
        epoch_losses = {'G_Total': 0.0, 'D_Total': 0.0, 'G_L1': 0.0, 'G_GAN': 0.0, 'G_VGG': 0.0, 'PSNR': 0.0}
        num_batches = 0

        print(f'\n🔵 Epoch {epoch}/{end_epoch} 开始...')

        for i, data in enumerate(dataset):
            iter_start_time = time.time()
            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time

            total_iters += opt.batch_size
            epoch_iter += opt.batch_size
            
            # --- 核心训练步 ---
            model.set_input(data)
            model.optimize_parameters()
            
            # --- 收集 Loss 数据 ---
            losses = model.get_current_losses()
            # 注意：这里的 key 要和你 augan_model.py 里定义的 loss_names 对应
            # 通常是 G_GAN, G_Pixel(即L1), D_Real, D_Fake
            # 为了通用性，我这里做一个映射尝试，如果取不到就填 0
            g_loss = losses.get('G_GAN', 0) + losses.get('G_Pixel', 0) + losses.get('G_Perceptual', 0)
            d_loss = losses.get('D_Real', 0) + losses.get('D_Fake', 0)
            
            epoch_losses['G_Total'] += g_loss
            epoch_losses['D_Total'] += d_loss
            epoch_losses['G_L1']    += losses.get('G_Pixel', 0)
            epoch_losses['G_GAN']   += losses.get('G_GAN', 0)
            epoch_losses['G_VGG']   += losses.get('G_Perceptual', 0)

            # --- 计算 Training PSNR (仅供参考) ---
            # 获取当前 batch 的图像
            model.compute_visuals()
            visuals = model.get_current_visuals()
            # visuals 里通常有 real_lq, fake_hq, real_sq (对应 augan_model.py 的 visual_names)
            # 如果名字不一样，这里会自动 fallback
            fake_im = tensor2im_custom(visuals.get('fake_hq', list(visuals.values())[0]))
            real_im = tensor2im_custom(visuals.get('real_sq', list(visuals.values())[1]))
            epoch_losses['PSNR'] += calculate_psnr(fake_im, real_im)

            num_batches += 1

            # --- 屏幕打印 (Print Freq) ---
            if total_iters % opt.print_freq == 0:
                t_comp = (time.time() - iter_start_time) / opt.batch_size
                print(f"Epoch: {epoch} | Iters: {epoch_iter} | Time: {t_comp:.3f}s | "
                      f"G_L1: {losses.get('G_Pixel', 0):.4f} | G_GAN: {losses.get('G_GAN', 0):.4f}")

            # --- 保存最新模型 (Freq) ---
            if total_iters % opt.save_latest_freq == 0:
                print(f'💾 保存 latest 模型 (epoch {epoch}, iters {total_iters})')
                model.save_networks('latest')

            iter_data_time = time.time()

        # =================================================
        # End of Epoch: 统计、日志、绘图
        # =================================================
        
        # 1. 计算平均值
        for k in epoch_losses:
            epoch_losses[k] /= max(num_batches, 1)
        
        time_taken = time.time() - epoch_start_time
        current_lr = model.optimizers[0].param_groups[0]['lr']

        # 2. 写入 CSV
        with open(log_name, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch, 
                             f"{time_taken:.1f}", 
                             f"{epoch_losses['G_Total']:.4f}", 
                             f"{epoch_losses['D_Total']:.4f}", 
                             f"{epoch_losses['G_L1']:.4f}", 
                             f"{epoch_losses['G_GAN']:.4f}", 
                             f"{epoch_losses['G_VGG']:.4f}", 
                             f"{epoch_losses['PSNR']:.2f}", 
                             f"{current_lr:.6f}"])
        print(f"✅ Epoch {epoch} 结束. Time: {time_taken:.1f}s, Avg PSNR: {epoch_losses['PSNR']:.2f} dB")

        # 3. [增强功能 2] 生成对比图 (Input | Fake | Truth)
        # 取本 Epoch 最后一个 Batch 的数据来画图
        visuals = model.get_current_visuals()
        
        # 提取图片 (确保 key 与 augan_model.py 一致)
        # 你的 dataset 返回的是 lq, sq. augan_model set_input 应该映射为了 real_lq, real_sq
        img_lq = tensor2im_custom(visuals.get('real_lq'))
        img_fake = tensor2im_custom(visuals.get('fake_hq'))
        img_sq = tensor2im_custom(visuals.get('real_sq'))
        
        # 拼图 (横向拼接)
        # 形状: [H, W, 3]
        h, w, c = img_lq.shape
        combined_image = np.concatenate([img_lq, img_fake, img_sq], axis=1) # 1 是宽度方向
        
        # 保存
        img_dir = os.path.join(expr_dir, 'web_images')
        if not os.path.exists(img_dir):
            os.makedirs(img_dir)
        
        save_path = os.path.join(img_dir, f'epoch_{epoch:03d}_comparison.png')
        Image.fromarray(combined_image).save(save_path)
        print(f"🖼️ 保存对比图: {save_path}")

        # 4. 保存模型 (按 Epoch)
        if epoch % opt.save_epoch_freq == 0:
            print(f'💾 保存 epoch {epoch} 模型...')
            model.save_networks('latest')
            model.save_networks(epoch)

    print("🏁 所有训练完成!")

# """
# AUGAN 3D 训练主入口脚本 (V9.5 - 5图可视化修复版)
# 修改说明：
# 1. 修复可视化：现在调用 util.save_training_images，显示 5 张图 (含 Diff Map)。
# 2. 保留 CSV 导出功能。
# """
# import time
# import os
# import torch
# import numpy as np
# import random
# import csv
# from tqdm import tqdm
# from torch.utils.tensorboard import SummaryWriter
# import matplotlib
# matplotlib.use('Agg') 
# import matplotlib.pyplot as plt
# import nibabel as nib 

# # [关键] 导入高级可视化函数
# from util import save_training_images

# try:
#     from options.train_options import TrainOptions
# except ImportError:
#     import sys
#     sys.path.append('.')
#     from options.train_options import TrainOptions

# from data import create_dataset
# from models import create_model

# def set_seed(seed):
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)
#     np.random.seed(seed)
#     random.seed(seed)
#     torch.backends.cudnn.deterministic = True

# def print_training_summary(opt, dataset, model):
#     device = torch.device('cuda:{}'.format(opt.gpu_ids[0])) if opt.gpu_ids else torch.device('cpu')
#     print("\n" + "="*80)
#     print(f"{'🚀 AUGAN TRAINING CONFIGURATION':^80}")
#     print("="*80)
#     print(f"  - Device:        {device}")
#     print(f"  - Data Root:     {opt.dataroot}")
#     print(f"  - Dataset Size:  {len(dataset)} volumes")
#     print(f"  - Batch Size:    {opt.batch_size}")
#     print(f"  - Model:         G={opt.netG}, D={opt.netD}")
#     print("="*80 + "\n")

# def print_epoch_report(epoch, total_epochs, epoch_time, losses_avg, lr_G, lr_D):
#     print('-' * 80)
#     print(f'END OF EPOCH {epoch} / {total_epochs} \t Time Taken: {epoch_time:.0f} sec')
#     print(f'  Learning Rates: \t G_lr = {lr_G:.7f} | D_lr = {lr_D:.7f}')
#     for k, v in losses_avg.items():
#         if 'G_' in k: print(f'      {k}: \t {v:.4f}')
#     print('-' * 80 + '\n')

# # ==============================================================================
# # [主程序]
# # ==============================================================================
# if __name__ == '__main__':
#     opt_driver = TrainOptions() 
#     opt = opt_driver.parse()    
#     set_seed(42)
    
#     log_dir = os.path.join(opt.checkpoints_dir, opt.name, 'logs')
#     img_save_dir = os.path.join(opt.checkpoints_dir, opt.name, 'web_images')
#     os.makedirs(log_dir, exist_ok=True)
#     os.makedirs(img_save_dir, exist_ok=True)
    
#     csv_log_path = os.path.join(opt.checkpoints_dir, opt.name, 'loss_log.csv')
#     writer = SummaryWriter(log_dir=log_dir)

#     dataset = create_dataset(opt)
#     model = create_model(opt)
#     model.setup(opt)
    
#     print_training_summary(opt, dataset, model)
    
#     total_iters = 0                
#     total_epochs = opt.n_epochs + opt.n_epochs_decay
    
#     print("📸 Saving initial sample (Step 0 check)...")
#     init_batch = next(iter(dataset))
#     model.set_input(init_batch)
#     model.forward()
    
#     # [修复] 调用高级可视化 (显示 5 张图)
#     save_training_images(
#         model.real_lq, model.fake_hq, model.real_hq, model.real_sq, 
#         0, img_save_dir, dynamic_range=60
#     )
    
#     # --- 训练循环 ---
#     for epoch in range(opt.epoch_count, total_epochs + 1):
#         epoch_start_time = time.time()
#         epoch_losses = {} 
#         epoch_iter_count = 0
        
#         print(f"\nStart Epoch {epoch} / {total_epochs}")
#         progress_bar = tqdm(dataset, desc=f"Epoch {epoch}", unit="batch")

#         for i, data in enumerate(progress_bar):
#             total_iters += opt.batch_size
#             epoch_iter_count += 1
            
#             model.set_input(data)         
#             model.optimize_parameters()   
            
#             current_losses = model.get_current_losses()
#             for k, v in current_losses.items():
#                 epoch_losses[k] = epoch_losses.get(k, 0.0) + v

#             if total_iters % opt.print_freq == 0:    
#                 loss_display = {k.replace('G_', ''): f"{v:.3f}" for k, v in current_losses.items() if 'G_' in k}
#                 progress_bar.set_postfix(**loss_display)
#                 for k, v in current_losses.items():
#                     writer.add_scalar(f'Loss_Step/{k}', v, total_iters)
                    
#         # 计算平均 Loss
#         avg_losses = {k: v / epoch_iter_count for k, v in epoch_losses.items()}
#         for k, v in avg_losses.items():
#             writer.add_scalar(f'Loss_Epoch/{k}', v, epoch)
            
#         lr_G = model.optimizers[0].param_groups[0]['lr']
#         lr_D = model.optimizers[1].param_groups[0]['lr']
        
#         print_epoch_report(epoch, total_epochs, time.time() - epoch_start_time, avg_losses, lr_G, lr_D)
        
#         # --- CSV 保存 ---
#         try:
#             write_header = not os.path.exists(csv_log_path)
#             with open(csv_log_path, mode='a', newline='') as f:
#                 fieldnames = ['epoch'] + sorted(avg_losses.keys())
#                 writer_csv = csv.DictWriter(f, fieldnames=fieldnames)
#                 if write_header: writer_csv.writeheader()
#                 row = {'epoch': epoch}; row.update(avg_losses)
#                 writer_csv.writerow(row)
#             print(f"  📈 CSV Log Saved")
#         except Exception as e:
#             print(f"  ⚠️ CSV Error: {e}")

#         # --- [修复] 可视化 ---
#         save_training_images(
#             model.real_lq, model.fake_hq, model.real_hq, model.real_sq, 
#             epoch, img_save_dir, dynamic_range=60
#         )
        
#         if epoch % opt.save_epoch_freq == 0:
#             print(f'💾 Saving checkpoints at epoch {epoch}')
#             model.save_networks('latest')
#             model.save_networks(epoch)

#         model.update_learning_rate() 
        
#     writer.close()
#     print("🎉 Training Finished!")