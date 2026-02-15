import torch
import torch.nn as nn
from .losses import TVLoss, FrequencyLoss, VGG19FeatureExtractor
from utils.ssim_loss import SSIMLoss
from . import networks

class LossHelper(nn.Module):
    """
    [新增组件] LossHelper
    职责:
    1. 集中管理所有 Loss 的初始化 (GAN, L1, TV, Freq, VGG, SSIM)
    2. 自动处理 3D 数据的维度变形 (Reshape)，确保 2D Loss (SSIM/VGG) 不会报错
    3. 提供统一的计算接口 compute_G_loss
    """
    def __init__(self, opt, device):
        super(LossHelper, self).__init__()
        self.opt = opt
        self.device = device
        
        # 1. 初始化 Loss 函数
        # -----------------------------------------------------------
        # GAN Loss (自动判断 use_lsgan)
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
        返回: (total_loss, loss_dict)
        """
        loss_dict = {}
        total_loss = 0.0
        
        # 1. GAN Loss
        # -----------------------------------------------------------
        fake_AB = torch.cat((input_lq, fake_sq), 1)
        pred_fake = netD(fake_AB)
        loss_G_GAN = self.criterionGAN(pred_fake, True) * self.opt.lambda_gan
        
        loss_dict['G_GAN'] = loss_G_GAN
        total_loss += loss_G_GAN
        
        # 2. Pixel Loss (L1)
        # -----------------------------------------------------------
        loss_G_L1 = self.criterionL1(fake_sq, real_sq) * self.opt.lambda_pixel
        
        loss_dict['G_L1'] = loss_G_L1
        total_loss += loss_G_L1
        
        # 3. TV Loss
        # -----------------------------------------------------------
        if self.criterionTV:
            loss_TV = self.criterionTV(fake_sq) * self.opt.lambda_tv
            loss_dict['G_TV'] = loss_TV
            total_loss += loss_TV
            
        # 4. Frequency Loss
        # -----------------------------------------------------------
        if self.criterionFreq:
            loss_Freq = self.criterionFreq(fake_sq, real_sq) * self.opt.lambda_ffl
            loss_dict['G_Freq'] = loss_Freq
            total_loss += loss_Freq
            
        # 5. VGG Loss (修复：先提取特征，再算距离)
        # -----------------------------------------------------------
        if self.criterionVGG:
            # 预处理: 反归一化 + 维度变形
            fake_2d, real_2d = self._preprocess_for_2d_loss(fake_sq, real_sq, need_3channel=True)
            
            # 分别提取特征 (forward 只接受一个参数)
            fake_features = self.criterionVGG(fake_2d)
            real_features = self.criterionVGG(real_2d)
            
            # 计算 L1 距离 (兼容列表返回或单张量返回)
            loss_VGG = 0
            if isinstance(fake_features, list):
                # 如果是多层特征
                for f, r in zip(fake_features, real_features):
                    loss_VGG += self.criterionL1(f, r)
            else:
                # 如果是单层特征
                loss_VGG = self.criterionL1(fake_features, real_features)
            
            loss_VGG = loss_VGG * self.opt.lambda_perceptual
            
            loss_dict['G_VGG'] = loss_VGG
            total_loss += loss_VGG

        # 6. SSIM Loss (自动处理 3D -> 2D)
        # -----------------------------------------------------------
        if self.criterionSSIM:
            # 预处理: 反归一化 + 维度变形 (SSIM不需要强制3通道，单通道即可)
            fake_2d, real_2d = self._preprocess_for_2d_loss(fake_sq, real_sq, need_3channel=False)
            loss_SSIM = self.criterionSSIM(fake_2d, real_2d) * self.opt.lambda_ssim
            
            loss_dict['G_SSIM'] = loss_SSIM
            total_loss += loss_SSIM
            
        return total_loss, loss_dict

    def compute_D_loss(self, netD, fake_sq, real_sq, input_lq):
        """
        计算 Discriminator 的 Loss
        """
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
        1. 反归一化 [-1, 1] -> [0, 1]
        2. 如果是 3D (B,C,D,H,W) -> Reshape 为 (B*D, C, H, W)
        3. 如果需要 3通道 -> Repeat
        """
        # 1. Denorm
        fake_01 = (fake + 1.0) / 2.0
        real_01 = (real + 1.0) / 2.0
        
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