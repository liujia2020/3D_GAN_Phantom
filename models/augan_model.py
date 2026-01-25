import torch
import torch.nn as nn
from .base_model import BaseModel
from . import networks
import torchvision.models as models
import torch.nn.functional as F
from utils.ssim_loss import SSIMLoss
# =========================================================================
# [模块 1] VGG19 特征提取器
# 用于计算感知损失 (Perceptual Loss)，提取图像的纹理和形状特征
# =========================================================================
class VGG19FeatureExtractor(nn.Module):
    def __init__(self):
        super(VGG19FeatureExtractor, self).__init__()
        # 加载预训练的 VGG19 网络
        vgg19 = models.vgg19(pretrained=True)
        # 截取 VGG 的前 30 层 (到 relu5_2 附近)，包含丰富的纹理和形状信息
        self.features = nn.Sequential(*list(vgg19.features.children())[:30])
        
        # 冻结参数，不参与训练
        for param in self.features.parameters():
            param.requires_grad = False
        
    def forward(self, x):
        return self.features(x)

# =========================================================================
# [模块 2] Total Variation Loss (TV Loss)
# 用于平滑图像，消除高频伪影（如网格线、毛刺、海胆刺）
# =========================================================================
class TVLoss(nn.Module):
    def __init__(self, weight=1.0):
        super(TVLoss, self).__init__()
        self.weight = weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self.tensor_size(x[:, :, 1:, :])
        count_w = self.tensor_size(x[:, :, :, 1:])
        
        # 计算水平和垂直方向的梯度差 (L2 范数)
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        return self.weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size

    def tensor_size(self, t):
        return t.size()[1] * t.size()[2] * t.size()[3]

# =========================================================================
# [主类] Augan Model
# =========================================================================
class AuganModel(BaseModel):
    def modify_commandline_options(parser, is_train=True):
        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)
        # 定义 Loss 名称，用于日志打印
        # 注意：只有权重 > 0 的 Loss 才会真正计算，但这里我们列出所有可能的 Loss
        self.loss_names = ['G_GAN', 'G_Pixel', 'G_Edge', 'G_Perceptual', 'G_TV', 'G_SSIM','D_Real', 'D_Fake']
        # self.loss_names = ['G_GAN', 'G_Pixel','D_Real', 'D_Fake']
        # 定义可视化图片
        self.visual_names = ['real_lq', 'fake_hq', 'real_sq'] 
        
        if self.isTrain:
            self.model_names = ['G', 'D']
        else:
            self.model_names = ['G']

        # 定义生成器
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:
            # 定义判别器
            self.netD = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf, opt.netD,
                                          n_layers_D=3, norm=opt.norm, init_type=opt.init_type, init_gain=opt.init_gain, gpu_ids=self.gpu_ids)

            # --- 定义 Loss 函数 ---
            # 1. GAN Loss
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            # 2. Pixel Loss
            self.criterionPixel = torch.nn.L1Loss() 
            # 3. TV Loss (仅初始化，具体是否计算取决于 lambda_tv)
            self.criterionTV = TVLoss(weight=1.0).to(self.device)
            # 4. Perceptual Loss (仅当需要时初始化，节省显存)
            if opt.lambda_perceptual > 0:
                self.netVGG = VGG19FeatureExtractor().to(self.device)
                self.criterionPerceptual = torch.nn.L1Loss()
            if opt.lambda_ssim > 0:
                self.criterionSSIM = SSIMLoss().to(self.device)
            # --- 优化器 ---
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr * opt.lr_d_ratio, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

    def set_input(self, input):
        self.real_lq = input['lq'].to(self.device)
        self.real_sq = input['sq'].to(self.device)
        self.real_hq = input['hq'].to(self.device)
        self.image_paths = input['case_name']

    def forward(self):
        # 把输入送到G，得到fake输出。
        self.fake_hq = self.netG(self.real_lq)

    def backward_D(self):
        # Fake 判别器得到生成的图，认为是真图的概率。越小越好，意味着判别器认为生成图是假的。输出接近0.
        # 如果判别器被骗了，认为生成图是真的，loss就会变大。
        fake_AB = torch.cat((self.real_lq, self.fake_hq), 1)
        pred_fake = self.netD(fake_AB.detach())
        self.loss_D_Fake = self.criterionGAN(pred_fake, False)

        # Real 判别器得到GT，认定为真图的概率。越小越好。1为真。
        # 如果判别器认为GT是假的，loss就会变大。
        real_AB = torch.cat((self.real_lq, self.real_sq), 1)
        pred_real = self.netD(real_AB)
        self.loss_D_Real = self.criterionGAN(pred_real, True)

        # Combined Loss
        self.loss_D = (self.loss_D_Fake + self.loss_D_Real) * 0.5
        self.loss_D.backward()

    def backward_G(self):
        # 1. GAN Loss
        fake_AB = torch.cat((self.real_lq, self.fake_hq), 1)
        pred_fake = self.netD(fake_AB)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True)

        # 2. Pixel Loss
        if self.opt.lambda_pixel > 0:
            self.loss_G_Pixel = self.criterionPixel(self.fake_hq, self.real_sq) * self.opt.lambda_pixel
        else:
            self.loss_G_Pixel = torch.tensor(0.0).to(self.device)
        
        # 3. Edge Loss
        if self.opt.lambda_edge > 0:
            self.loss_G_Edge = self.compute_edge_loss(self.fake_hq, self.real_sq) * self.opt.lambda_edge
        else:
            self.loss_G_Edge = torch.tensor(0.0).to(self.device)

        # 4. Perceptual Loss (3D 切片版)
        if self.opt.lambda_perceptual > 0:
            # Reshape: [B, C, D, H, W] -> [B*D, C, H, W]
            b, c, d, h, w = self.fake_hq.shape
            fake_2d = self.fake_hq.permute(0, 2, 1, 3, 4).contiguous().view(b * d, c, h, w)
            real_2d = self.real_sq.permute(0, 2, 1, 3, 4).contiguous().view(b * d, c, h, w)
            
            # VGG 输入必须是 3 通道，如果是单通道则复制
            if c == 1:
                fake_2d = fake_2d.repeat(1, 3, 1, 1)
                real_2d = real_2d.repeat(1, 3, 1, 1)
            
            feat_fake = self.netVGG(fake_2d)
            with torch.no_grad():
                feat_real = self.netVGG(real_2d)
            
            self.loss_G_Perceptual = self.criterionPerceptual(feat_fake, feat_real) * self.opt.lambda_perceptual
        else:
            self.loss_G_Perceptual = torch.tensor(0.0).to(self.device)

        # 5. TV Loss (去伪影/去海胆刺)
        if self.opt.lambda_tv > 0:
            # 同样将 3D 视为多个 2D 切片计算
            b, c, d, h, w = self.fake_hq.shape
            fake_for_tv = self.fake_hq.permute(0, 2, 1, 3, 4).contiguous().view(b * d, c, h, w)
            self.loss_G_TV = self.criterionTV(fake_for_tv) * self.opt.lambda_tv
        else:
            self.loss_G_TV = torch.tensor(0.0).to(self.device)

        # [新增] 6. SSIM Loss
        if self.opt.lambda_ssim > 0:
            # 同样需要把 3D [B, C, D, H, W] 转为 2D [B*D, C, H, W] 进行计算
            b, c, d, h, w = self.fake_hq.shape
            fake_2d = self.fake_hq.permute(0, 2, 1, 3, 4).contiguous().view(b * d, c, h, w)
            real_2d = self.real_sq.permute(0, 2, 1, 3, 4).contiguous().view(b * d, c, h, w)
            
            self.loss_G_SSIM = self.criterionSSIM(fake_2d, real_2d) * self.opt.lambda_ssim
        else:
            self.loss_G_SSIM = torch.tensor(0.0).to(self.device)
        
        
        # 总 Loss 求和
        self.loss_G = self.loss_G_GAN + self.loss_G_Pixel + self.loss_G_Edge + self.loss_G_Perceptual + self.loss_G_TV + self.loss_G_SSIM
        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()
        # 更新判别器
        self.set_requires_grad(self.netD, True)
        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step()
        # 更新生成器
        self.set_requires_grad(self.netD, False)
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()
        
    def compute_edge_loss(self, fake, real):
        # 简单的梯度 L1 Loss
        def gradient(x):
            # padding 保证尺寸一致
            r = F.pad(x, (0, 1, 0, 0, 0, 0))[:, :, :, :, 1:] - x
            l = F.pad(x, (1, 0, 0, 0, 0, 0))[:, :, :, :, :-1] - x
            t = F.pad(x, (0, 0, 1, 0, 0, 0))[:, :, :, :-1, :] - x
            b = F.pad(x, (0, 0, 0, 1, 0, 0))[:, :, :, 1:, :] - x
            f = F.pad(x, (0, 0, 0, 0, 1, 0))[:, :, :-1, :, :] - x
            bk = F.pad(x, (0, 0, 0, 0, 0, 1))[:, :, 1:, :, :] - x
            return torch.abs(r) + torch.abs(l) + torch.abs(t) + torch.abs(b) + torch.abs(f) + torch.abs(bk)

        return torch.nn.L1Loss()(gradient(fake), gradient(real))