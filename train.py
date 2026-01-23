"""
AUGAN 3D 训练主入口脚本 (V9.5 - 5图可视化修复版)
修改说明：
1. 修复可视化：现在调用 util.save_training_images，显示 5 张图 (含 Diff Map)。
2. 保留 CSV 导出功能。
"""
import time
import os
import torch
import numpy as np
import random
import csv
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import nibabel as nib 

# [关键] 导入高级可视化函数
from util import save_training_images

try:
    from options.train_options import TrainOptions
except ImportError:
    import sys
    sys.path.append('.')
    from options.train_options import TrainOptions

from data import create_dataset
from models import create_model

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def print_training_summary(opt, dataset, model):
    device = torch.device('cuda:{}'.format(opt.gpu_ids[0])) if opt.gpu_ids else torch.device('cpu')
    print("\n" + "="*80)
    print(f"{'🚀 AUGAN TRAINING CONFIGURATION':^80}")
    print("="*80)
    print(f"  - Device:        {device}")
    print(f"  - Data Root:     {opt.dataroot}")
    print(f"  - Dataset Size:  {len(dataset)} volumes")
    print(f"  - Batch Size:    {opt.batch_size}")
    print(f"  - Model:         G={opt.netG}, D={opt.netD}")
    print("="*80 + "\n")

def print_epoch_report(epoch, total_epochs, epoch_time, losses_avg, lr_G, lr_D):
    print('-' * 80)
    print(f'END OF EPOCH {epoch} / {total_epochs} \t Time Taken: {epoch_time:.0f} sec')
    print(f'  Learning Rates: \t G_lr = {lr_G:.7f} | D_lr = {lr_D:.7f}')
    for k, v in losses_avg.items():
        if 'G_' in k: print(f'      {k}: \t {v:.4f}')
    print('-' * 80 + '\n')

# ==============================================================================
# [主程序]
# ==============================================================================
if __name__ == '__main__':
    opt_driver = TrainOptions() 
    opt = opt_driver.parse()    
    set_seed(42)
    
    log_dir = os.path.join(opt.checkpoints_dir, opt.name, 'logs')
    img_save_dir = os.path.join(opt.checkpoints_dir, opt.name, 'web_images')
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(img_save_dir, exist_ok=True)
    
    csv_log_path = os.path.join(opt.checkpoints_dir, opt.name, 'loss_log.csv')
    writer = SummaryWriter(log_dir=log_dir)

    dataset = create_dataset(opt)
    model = create_model(opt)
    model.setup(opt)
    
    print_training_summary(opt, dataset, model)
    
    total_iters = 0                
    total_epochs = opt.n_epochs + opt.n_epochs_decay
    
    print("📸 Saving initial sample (Step 0 check)...")
    init_batch = next(iter(dataset))
    model.set_input(init_batch)
    model.forward()
    
    # [修复] 调用高级可视化 (显示 5 张图)
    save_training_images(
        model.real_lq, model.fake_hq, model.real_hq, model.real_sq, 
        0, img_save_dir, dynamic_range=60
    )
    
    # --- 训练循环 ---
    for epoch in range(opt.epoch_count, total_epochs + 1):
        epoch_start_time = time.time()
        epoch_losses = {} 
        epoch_iter_count = 0
        
        print(f"\nStart Epoch {epoch} / {total_epochs}")
        progress_bar = tqdm(dataset, desc=f"Epoch {epoch}", unit="batch")

        for i, data in enumerate(progress_bar):
            total_iters += opt.batch_size
            epoch_iter_count += 1
            
            model.set_input(data)         
            model.optimize_parameters()   
            
            current_losses = model.get_current_losses()
            for k, v in current_losses.items():
                epoch_losses[k] = epoch_losses.get(k, 0.0) + v

            if total_iters % opt.print_freq == 0:    
                loss_display = {k.replace('G_', ''): f"{v:.3f}" for k, v in current_losses.items() if 'G_' in k}
                progress_bar.set_postfix(**loss_display)
                for k, v in current_losses.items():
                    writer.add_scalar(f'Loss_Step/{k}', v, total_iters)
                    
        # 计算平均 Loss
        avg_losses = {k: v / epoch_iter_count for k, v in epoch_losses.items()}
        for k, v in avg_losses.items():
            writer.add_scalar(f'Loss_Epoch/{k}', v, epoch)
            
        lr_G = model.optimizers[0].param_groups[0]['lr']
        lr_D = model.optimizers[1].param_groups[0]['lr']
        
        print_epoch_report(epoch, total_epochs, time.time() - epoch_start_time, avg_losses, lr_G, lr_D)
        
        # --- CSV 保存 ---
        try:
            write_header = not os.path.exists(csv_log_path)
            with open(csv_log_path, mode='a', newline='') as f:
                fieldnames = ['epoch'] + sorted(avg_losses.keys())
                writer_csv = csv.DictWriter(f, fieldnames=fieldnames)
                if write_header: writer_csv.writeheader()
                row = {'epoch': epoch}; row.update(avg_losses)
                writer_csv.writerow(row)
            print(f"  📈 CSV Log Saved")
        except Exception as e:
            print(f"  ⚠️ CSV Error: {e}")

        # --- [修复] 可视化 ---
        save_training_images(
            model.real_lq, model.fake_hq, model.real_hq, model.real_sq, 
            epoch, img_save_dir, dynamic_range=60
        )
        
        if epoch % opt.save_epoch_freq == 0:
            print(f'💾 Saving checkpoints at epoch {epoch}')
            model.save_networks('latest')
            model.save_networks(epoch)

        model.update_learning_rate() 
        
    writer.close()
    print("🎉 Training Finished!")