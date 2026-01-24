import os
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.util import save_nii  # 假设 util 里有这个函数，或者你自己写的
import torch
import numpy as np
import nibabel as nib

def save_nii_custom(data, path):
    # 简易保存函数，防止 util 里没有
    data = data.squeeze().cpu().numpy()
    # 逆归一化 (可选，看你是否需要还原回 dB)
    # data = (data / 2.0 + 0.5) * 60 - 60
    
    # 确保保存维度顺序 (如果是 nibabel 习惯 x,y,z)
    if len(data.shape) == 3:
        data = data.transpose(2, 1, 0) # D,H,W -> W,H,D for ITK-SNAP
        
    new_image = nib.Nifti1Image(data, affine=np.eye(4))
    nib.save(new_image, path)

if __name__ == '__main__':
    opt = TestOptions().parse()  # get test options
    # hardcode some parameters for test
    opt.num_threads = 0   # test code only supports num_threads = 0
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True    # no flip for test

    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers

    if opt.eval:
        model.eval()
        
    print(f"Testing experiment: {opt.name}")
    
    for i, data in enumerate(dataset):
        if i >= opt.num_test:  # only apply our model to opt.num_test images.
            break
            
        model.set_input(data)  # unpack data from data loader
        model.test()           # run inference
        visuals = model.get_current_visuals()  # get image results
        
        img_path = model.get_image_paths()     # get image paths
        short_path = os.path.basename(img_path[0])
        name = os.path.splitext(short_path)[0]
        
        # === [核心修改] 文件名加上 opt.name ===
        # 原来: Case001_fake.nii
        # 现在: Case001_fake_experiment_v1.nii
        
        fake_im = visuals['fake_hq'] # 假设你的模型输出叫 fake_hq
        
        # 结果目录
        res_dir = os.path.join(opt.results_dir, opt.name)
        if not os.path.exists(res_dir):
            os.makedirs(res_dir)
            
        save_name = f"{name}_fake_{opt.name}.nii"
        save_path = os.path.join(res_dir, save_name)
        
        print(f"Saving {save_path} ...")
        save_nii_custom(fake_im, save_path)

# import os
# import torch
# import numpy as np
# import matplotlib
# matplotlib.use('Agg') 
# import matplotlib.pyplot as plt
# import nibabel as nib
# from tqdm import tqdm
# import pandas as pd
# import json
# import logging

# from options.test_options import TestOptions
# from data import create_dataset
# from models import create_model
# from utils.metrics import calc_metrics

# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# def save_visual_matrix_png(lq_avg, fake, hq, sq, case_name, save_dir, opt):
#     """
#     V4.5 核心：生成 4行x3列 的全景三视图矩阵 (LQ, Fake, HQ, SQ)
#     并自动校正物理长宽比
#     """
#     os.makedirs(os.path.join(save_dir, 'images'), exist_ok=True)
    
#     # 准备数据字典
#     data_map = {
#         'Input (LQ)': lq_avg,
#         'Generated (Fake)': fake,
#         'Ref (HQ)': hq,
#         'Target (SQ)': sq
#     }
    
#     # 获取中心切片索引
#     d_mid, h_mid, w_mid = lq_avg.shape[0] // 2, lq_avg.shape[1] // 2, lq_avg.shape[2] // 2
    
#     # 设置绘图布局: 4行 (数据) x 3列 (视图)
#     fig, axes = plt.subplots(4, 3, figsize=(15, 20), constrained_layout=True)
    
#     # 计算长轴的显示比例 (防止图像被拉伸成面条)
#     # 像素比例是 0.0362 / 0.2 ≈ 0.181
#     aspect_ratio_long = opt.spacing_z / opt.spacing_x 
    
#     for row_idx, (name, data) in enumerate(data_map.items()):
#         # 1. Axial View (横截面): Slice Dim 0 (D)
#         ax_axial = axes[row_idx, 0]
#         slice_axial = data[d_mid, :, :]
#         # 对数增强显示
#         img_ax = 20 * np.log10(np.maximum(slice_axial, 1e-6))
#         ax_axial.imshow(img_ax, cmap='gray', aspect='equal')
#         if row_idx == 0: ax_axial.set_title("Axial (H-W)\nCross-Section", fontsize=14, fontweight='bold')
#         ax_axial.set_ylabel(name, fontsize=16, fontweight='bold')
#         ax_axial.set_xticks([])
#         ax_axial.set_yticks([])

#         # 2. Coronal View (冠状面): Slice Dim 1 (H)
#         ax_cor = axes[row_idx, 1]
#         slice_cor = data[:, h_mid, :] 
#         img_cor = 20 * np.log10(np.maximum(slice_cor, 1e-6))
#         ax_cor.imshow(img_cor, cmap='gray', aspect=aspect_ratio_long)
#         if row_idx == 0: ax_cor.set_title("Coronal (D-W)\nLongitudinal", fontsize=14, fontweight='bold')
#         ax_cor.set_xticks([])
#         ax_cor.set_yticks([])

#         # 3. Sagittal View (矢状面): Slice Dim 2 (W)
#         ax_sag = axes[row_idx, 2]
#         slice_sag = data[:, :, w_mid]
#         img_sag = 20 * np.log10(np.maximum(slice_sag, 1e-6))
#         ax_sag.imshow(img_sag, cmap='gray', aspect=aspect_ratio_long)
#         if row_idx == 0: ax_sag.set_title("Sagittal (D-H)\nLongitudinal", fontsize=14, fontweight='bold')
#         ax_sag.set_xticks([])
#         ax_sag.set_yticks([])

#     # 保存高清大图
#     save_path = os.path.join(save_dir, 'images', f'{case_name}_Matrix_View.png')
#     plt.savefig(save_path, dpi=150)
#     plt.close(fig)
#     logging.info(f"三视图矩阵已保存: {save_path}")

# def save_full_volume_nifti_cloned(data, template_path, name, suffix, save_dir):
#     """V4.3: 克隆模板元数据保存 NIfTI"""
#     os.makedirs(os.path.join(save_dir, 'nifti'), exist_ok=True)
#     try:
#         template_img = nib.load(template_path)
#         affine = template_img.affine
#         header = template_img.header
#     except:
#         affine = np.eye(4)
#         header = None
        
#     nii_img = nib.Nifti1Image(data, affine, header)
#     save_path = os.path.join(save_dir, 'nifti', f'{name}_{suffix}.nii')
#     nib.save(nii_img, save_path)
#     logging.info(f"NIfTI文件(方向已校正)已保存: {save_path}")

# def denormalize_output(normalized_data, max_val):
#     """
#     [修正] 使用固定的物理 Max 值进行反归一化
#     不再依赖外部 json，确保输出数值是真实的物理强度
#     """
#     data_0_1 = (normalized_data + 1.0) / 2.0
#     return data_0_1 * max_val

# # def full_inference_8gb(model, input_tensor):
# #     """V4.1: 512分段推理 (显存优化) - 保持不变"""
# #     _, _, D, H, W = input_tensor.shape
# #     mid = 512
# #     output_full = np.zeros((D, H, W), dtype=np.float32)
# #     model.netG.cuda()
# #     with torch.no_grad():
# #         # Part 1
# #         part1_in = input_tensor[:, :, :mid, :, :].cuda()
# #         output_full[:mid, :, :] = model.netG(part1_in).squeeze().cpu().numpy()
# #         del part1_in
# #         torch.cuda.empty_cache()
# #         # Part 2
# #         part2_in = input_tensor[:, :, mid:, :, :].cuda()
# #         output_full[mid:, :, :] = model.netG(part2_in).squeeze().cpu().numpy()
# #         del part2_in
# #         torch.cuda.empty_cache()
# #     return output_full


# def full_inference_overlap(model, input_tensor, overlap=64):
#     """
#     [V5.0 - 4060 8GB 专用版] 带重叠的切分推理
#     解决 "中间有一条线" 的边界伪影问题。
#     原理：多算 64 层 (Overlap) 作为 Padding，拼接时丢弃边缘。
#     """
#     b, c, D, H, W = input_tensor.shape
    
#     # 1. 显存不够时的自动策略：如果深度 < 600，直接跑 (8GB 应该能勉强吃下 600)
#     # 如果深度太深 (1024)，则启动切分
#     if D < 600:
#         model.netG.cuda()
#         with torch.no_grad():
#             output_full = model.netG(input_tensor.cuda()).squeeze().cpu().numpy()
#         return output_full

#     # 2. 准备输出容器
#     output_full = np.zeros((D, H, W), dtype=np.float32)
#     model.netG.cuda()
    
#     mid = D // 2 # 512
    
#     with torch.no_grad():
#         # ================= Part 1: Top Half =================
#         # 输入范围: 0 ~ (512 + overlap)
#         # 目的: 只要 0 ~ 512 的纯净数据
#         end_idx = min(D, mid + overlap)
#         part1_in = input_tensor[:, :, :end_idx, :, :].cuda()
        
#         # 推理
#         out1 = model.netG(part1_in).squeeze().cpu().numpy()
        
#         # 裁剪: 只要 [0:mid]
#         output_full[:mid, :, :] = out1[:mid, :, :]
        
#         # 清显存
#         del part1_in, out1
#         torch.cuda.empty_cache()

#         # ================= Part 2: Bottom Half =================
#         # 输入范围: (512 - overlap) ~ 1024
#         # 目的: 只要 512 ~ 1024 的纯净数据
#         start_idx = max(0, mid - overlap)
#         part2_in = input_tensor[:, :, start_idx:, :, :].cuda()
        
#         # 推理
#         out2 = model.netG(part2_in).squeeze().cpu().numpy()
        
#         # 裁剪: 这里的 out2 长度是 (1024 - start_idx)
#         # 也就是 (1024 - (512 - overlap)) = 512 + overlap
#         # 我们的有效数据是从 overlap 开始的，对应的绝对位置是 512
#         valid_start_rel = mid - start_idx # 理论上等于 overlap
        
#         output_full[mid:, :, :] = out2[valid_start_rel:, :, :]
        
#         # 清显存
#         del part2_in, out2
#         torch.cuda.empty_cache()
        
#     return output_full

# if __name__ == '__main__':
#     opt = TestOptions().parse()
#     opt.num_threads = 0
#     opt.batch_size = 1
    
#     model = create_model(opt)
#     model.setup(opt)
#     model.eval()
    
#     dataset = create_dataset(opt)
#     save_root = os.path.join(opt.results_dir, opt.name, f"epoch_{opt.epoch}")
#     os.makedirs(save_root, exist_ok=True)
    
#     logging.info(f"🚀 V4.6 修复版测试启动 | 参数传递修复 | 物理数值校正")
    
#     file_list = dataset.dataset.file_list
#     metrics_list = []

#     # 直接从 Dataset 获取正确的物理定标值
#     NORM_MAX_INPUT = dataset.dataset.NORM_MAX_INPUT    # 500,000.0
#     NORM_MAX_TARGET = dataset.dataset.NORM_MAX_TARGET  # 25,000,000.0

#     for i, files in enumerate(tqdm(file_list)):
#         case_name = files['case_name']
#         template_sq_path = files['p_sq']
        
#         # 1. 核心推理
#         # [关键修复] 参数顺序修正: path, slice_objs, max_val
#         # 这里的 slice(None) 表示读取整个 1024x128x128 数据
#         full_slice = (slice(None), slice(None), slice(None))
        
#         input_data = [dataset.dataset._read_and_process(files[k], full_slice, NORM_MAX_INPUT) 
#                       for k in ['p_in_n15', 'p_in_000', 'p_in_p15']]
        
#         real_lq_tensor = torch.from_numpy(np.stack(input_data, axis=0)).unsqueeze(0).float()
        
#         # 此时 real_lq_tensor 深度应该是 1024，inference 函数将正常工作
#         # fake_norm = full_inference_overlap(model, real_lq_tensor)
#         fake_norm = full_inference_overlap(model, real_lq_tensor, overlap=64)

#         # 2. 反归一化 (使用 2.5e7 的基准，确保输出是百万级数值)
#         fake_denorm = denormalize_output(fake_norm, NORM_MAX_TARGET)
        
#         # 真值也用同样的基准还原
#         sq_denorm = denormalize_output(dataset.dataset._read_and_process(files['p_sq'], full_slice, NORM_MAX_TARGET), NORM_MAX_TARGET)
#         hq_denorm = denormalize_output(dataset.dataset._read_and_process(files['p_hq'], full_slice, NORM_MAX_TARGET), NORM_MAX_TARGET)
        
#         # Input 也要还原 (注意 Input 的基准是 5e5)
#         lq_avg_norm = np.mean(np.stack(input_data, axis=0), axis=0)
#         lq_avg_denorm = denormalize_output(lq_avg_norm, NORM_MAX_INPUT)

#         # 3. 指标计算
#         m_sq = calc_metrics(torch.from_numpy(fake_denorm).unsqueeze(0).unsqueeze(0), torch.from_numpy(sq_denorm).unsqueeze(0).unsqueeze(0))
#         m_hq = calc_metrics(torch.from_numpy(fake_denorm).unsqueeze(0).unsqueeze(0), torch.from_numpy(hq_denorm).unsqueeze(0).unsqueeze(0))

#         # 4. 保存文件
#         save_full_volume_nifti_cloned(fake_denorm, template_sq_path, case_name, 'Fake', save_root)
        
#         # 5. 保存可视化 (使用还原后的物理数值，log显示会很清晰)
#         save_visual_matrix_png(lq_avg_denorm, fake_denorm, hq_denorm, sq_denorm, case_name, save_root, opt)
        
#         metrics_list.append({'Name': case_name, 'PSNR_SQ': m_sq['PSNR'], 'SSIM_SQ': m_sq['SSIM'], 'PSNR_HQ': m_hq['PSNR'], 'SSIM_HQ': m_hq['SSIM']})

#     pd.DataFrame(metrics_list).to_csv(os.path.join(save_root, 'metrics_final.csv'), index=False)
#     logging.info(f"✅ 全部完成！输出文件已生成在 {save_root}")