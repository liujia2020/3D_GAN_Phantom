import torch
import torch.nn as nn
import math
from .losses import TVLoss, FrequencyLoss, VGG19FeatureExtractor
from utils.ssim_loss import SSIMLoss
from . import networks

class LossHelper(nn.Module):
    """
    [Exp42 复刻版] LossHelper
    1. 增加了 BG Loss (背景抑制) 的计算逻辑
    2. 在 VGG/SSIM 预处理中增加了 clamp(0, 1)，防止 NaN
    """
    def __init__(self, opt, device):
        super(LossHelper, self).__init__()
        self.opt = opt
        self.device = device
        
        # 1. 初始化 Loss 函数
        use_lsgan = True
        if hasattr(opt, 'gan_mode'):
            use_lsgan = (opt.gan_mode == 'lsgan')
        elif hasattr(opt, 'no_lsgan'):
            use_lsgan = not opt.no_lsgan
            
        self.criterionGAN = networks.GANLoss(use_lsgan=use_lsgan).to(device)
        self.criterionL1 = nn.L1Loss()
        
        # 可选 Loss
        self.criterionTV = TVLoss().to(device) if opt.lambda_tv > 0 else None
        self.criterionFreq = FrequencyLoss().to(device) if opt.lambda_ffl > 0 else None
        self.criterionVGG = VGG19FeatureExtractor().to(device) if opt.lambda_perceptual > 0 else None
        self.criterionSSIM = SSIMLoss().to(device) if opt.lambda_ssim > 0 else None

    def compute_G_loss(self, netD, fake_sq, real_sq, input_lq):
        """
        计算 Generator 的所有 Loss
        """
        loss_dict = {}
        total_loss = 0.0
        
        # 1. GAN Loss
        fake_AB = torch.cat((input_lq, fake_sq), 1)
        pred_fake = netD(fake_AB)
        loss_G_GAN = self.criterionGAN(pred_fake, True) * self.opt.lambda_gan
        loss_dict['G_GAN'] = loss_G_GAN
        total_loss += loss_G_GAN
        
        # # 2. Pixel Loss (L1)
        # loss_G_L1 = self.criterionL1(fake_sq, real_sq) * self.opt.lambda_pixel
        # loss_dict['G_L1'] = loss_G_L1
        # total_loss += loss_G_L1
        
        # 2. Pixel Loss (L1) [引入深度加权机制]
        depth_mode = getattr(self.opt, 'depth_weight_mode', 'none')
        
        if depth_mode == 'none':
            # 如果没开开关，使用原版普通 L1，保证旧实验的绝对兼容
            loss_G_L1 = self.criterionL1(fake_sq, real_sq) * self.opt.lambda_pixel
        else:
            # 开启深度加权
            max_w = getattr(self.opt, 'depth_weight_max', 5.0)
            
            # 1. 计算绝对误差图
            abs_err = torch.abs(fake_sq - real_sq)
            
            # 2. 获取深度 D (假设 tensor shape 为 B, C, D, H, W)
            D = fake_sq.shape[2]
            d_idx = torch.arange(D, device=self.device, dtype=torch.float32)
            
            # 3. 生成 1D 权重曲线
            if depth_mode == 'linear':
                weights = 1.0 + (max_w - 1.0) * (d_idx / (D - 1))
            elif depth_mode == 'exp':
                k = math.log(max_w) / (D - 1)
                weights = torch.exp(k * d_idx)
            else:
                weights = torch.ones(D, device=self.device)
                
            # 4. 广播机制 (Broadcasting)：将 1D 变形为 (1, 1, D, 1, 1) 然后相乘
            weights = weights.view(1, 1, D, 1, 1)
            weighted_err = abs_err * weights
            
            # 5. 求全局平均，并乘上基础权重
            loss_G_L1 = torch.mean(weighted_err) * self.opt.lambda_pixel
            
        loss_dict['G_L1'] = loss_G_L1
        total_loss += loss_G_L1
        
        # 3. TV Loss
        if self.criterionTV:
            loss_TV = self.criterionTV(fake_sq) * self.opt.lambda_tv
            loss_dict['G_TV'] = loss_TV
            total_loss += loss_TV
            
        # 4. Frequency Loss
        if self.criterionFreq:
            loss_Freq = self.criterionFreq(fake_sq, real_sq) * self.opt.lambda_ffl
            loss_dict['G_Freq'] = loss_Freq
            total_loss += loss_Freq
            
        # 5. BG Loss (背景抑制) - [复活]
        # -----------------------------------------------------------
        if self.opt.lambda_bg > 0:
            # 逻辑: 找出 GT 中接近纯黑的区域 (< -0.9)，对该区域的 L1 误差加权惩罚
            # 假设数据归一化到 [-1, 1], 背景接近 -1
            bg_mask = (real_sq < -0.9).float().detach()
            
            # 计算掩码区域内的 L1 Loss
            # 公式: mean( abs(fake - real) * mask )
            loss_bg_raw = torch.mean(torch.abs(fake_sq - real_sq) * bg_mask)
            
            loss_BG = loss_bg_raw * self.opt.lambda_bg
            loss_dict['G_BG'] = loss_BG
            total_loss += loss_BG

        # 6. VGG Loss (特征感知)
        # -----------------------------------------------------------
        if self.criterionVGG:
            # 预处理: 反归一化 + Clamp + 维度变形
            fake_2d, real_2d = self._preprocess_for_2d_loss(fake_sq, real_sq, need_3channel=True)
            
            # 提取特征
            fake_features = self.criterionVGG(fake_2d)
            real_features = self.criterionVGG(real_2d)
            
            loss_VGG = 0
            if isinstance(fake_features, list):
                for f, r in zip(fake_features, real_features):
                    loss_VGG += self.criterionL1(f, r)
            else:
                loss_VGG = self.criterionL1(fake_features, real_features)
            
            loss_VGG = loss_VGG * self.opt.lambda_perceptual
            loss_dict['G_VGG'] = loss_VGG
            total_loss += loss_VGG

        # 7. SSIM Loss (结构相似度)
        # -----------------------------------------------------------
        if self.criterionSSIM:
            # 预处理
            fake_2d, real_2d = self._preprocess_for_2d_loss(fake_sq, real_sq, need_3channel=False)
            
            # Loss = 1 - SSIM (因为 SSIM 越大越好)
            loss_SSIM = (1.0 - self.criterionSSIM(fake_2d, real_2d)) * self.opt.lambda_ssim
            loss_dict['G_SSIM'] = loss_SSIM
            total_loss += loss_SSIM
            
        return total_loss, loss_dict

    def compute_D_loss(self, netD, fake_sq, real_sq, input_lq):
        # Fake
        fake_AB = torch.cat((input_lq, fake_sq), 1)
        pred_fake = netD(fake_AB.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        
        # Real
        real_AB = torch.cat((input_lq, real_sq), 1)
        pred_real = netD(real_AB)
        loss_D_real = self.criterionGAN(pred_real, True)
        
        loss_D = (loss_D_fake + loss_D_real) * 0.5
        return loss_D, {'D_real': loss_D_real, 'D_fake': loss_D_fake}

    def _preprocess_for_2d_loss(self, fake, real, need_3channel=False):
        """
        私有工具: 将数据转换为适合 2D Loss 的格式
        [关键修复]: 增加了 clamp(0.0, 1.0) 确保数值稳定
        """
        # 1. Denorm [-1, 1] -> [0, 1]
        fake_01 = (fake + 1.0) / 2.0
        real_01 = (real + 1.0) / 2.0
        
        # [安全锁] 确保数值绝对在 0-1 之间，防止微小误差导致 SSIM/VGG 出现 NaN
        fake_01 = torch.clamp(fake_01, 0.0, 1.0)
        real_01 = torch.clamp(real_01, 0.0, 1.0)
        
        # 2. Reshape 3D -> 2D
        if fake_01.ndim == 5:
            b, c, d, h, w = fake_01.shape
            fake_01 = fake_01.contiguous().view(b * d, c, h, w)
            real_01 = real_01.contiguous().view(b * d, c, h, w)
            
        # 3. Channel Repeat (for VGG)
        if need_3channel and fake_01.shape[1] == 1:
            fake_01 = fake_01.repeat(1, 3, 1, 1)
            real_01 = real_01.repeat(1, 3, 1, 1)
            
        return fake_01, real_01