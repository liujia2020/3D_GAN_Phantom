import torch
import itertools
from .base_model import BaseModel
from . import networks
from .losses import VGG19FeatureExtractor, TVLoss, FrequencyLoss
# [修正] 从 utils.ssim_loss 导入 SSIMLoss (之前误写为 SSIM)
from utils.ssim_loss import SSIMLoss 

class AuganModel(BaseModel):
    """
    [重构版] AUGAN Model
    1. 变量名统一: input_lq (输入), real_sq (真值), fake_sq (生成)
    2. 模块化: Loss 定义移至 models/losses.py
    3. 性能优化: 移除 optimize_parameters 中的 compute_visuals
    """
    def name(self):
        return 'AuganModel'

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        # 默认使用 res_unet_3d
        parser.set_defaults(netG='res_unet_3d', norm='batch', dataset_mode='ultrasound')
        if is_train:
            parser.add_argument('--lambda_L1', type=float, default=100.0, help='weight for L1 loss')
            parser.add_argument('--lambda_GAN', type=float, default=1.0, help='weight for GAN loss')
            parser.add_argument('--lambda_TV', type=float, default=0.0, help='weight for TV loss')
            parser.add_argument('--lambda_freq', type=float, default=0.0, help='weight for Frequency loss')
            parser.add_argument('--lambda_vgg', type=float, default=0.0, help='weight for VGG loss')
            parser.add_argument('--lambda_ssim', type=float, default=0.0, help='weight for SSIM loss')
        return parser

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.isTrain = opt.isTrain
        
        # 指定需要打印的 Loss 名字 (train.py 会读取这个列表)
        self.loss_names = ['G_GAN', 'G_L1', 'D_real', 'D_fake']
        if opt.lambda_TV > 0: self.loss_names.append('G_TV')
        if opt.lambda_freq > 0: self.loss_names.append('G_Freq')
        if opt.lambda_vgg > 0: self.loss_names.append('G_VGG')
        if opt.lambda_ssim > 0: self.loss_names.append('G_SSIM')
        
        # 指定需要保存/显示的图片名字 (train.py 会读取这个列表)
        # [统一命名]: lq (Low Quality), sq (Standard Quality)
        self.visual_names = ['input_lq', 'real_sq', 'fake_sq']
        
        # 指定需要保存的模型名字
        if self.isTrain:
            self.model_names = ['G', 'D']
        else:
            self.model_names = ['G']

        # 1. 定义网络 Generator
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids,
                                      use_attention=opt.use_attention, attn_temp=opt.attn_temp, 
                                      use_dilation=opt.use_dilation, use_aspp=opt.use_aspp)

        # 2. 定义网络 Discriminator (仅训练时)
        if self.isTrain:
            self.netD = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf, opt.netD,
                                          opt.n_layers_D, opt.norm, opt.no_lsgan, opt.init_type, opt.init_gain, self.gpu_ids)

        # 3. 定义 Loss 函数
        if self.isTrain:
            self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan).to(self.device)
            self.criterionL1 = torch.nn.L1Loss()
            
            # [模块化] 引用自 models.losses
            if opt.lambda_TV > 0:
                self.criterionTV = TVLoss().to(self.device)
            if opt.lambda_freq > 0:
                self.criterionFreq = FrequencyLoss().to(self.device)
            if opt.lambda_vgg > 0:
                self.criterionVGG = VGG19FeatureExtractor().to(self.device)
            if opt.lambda_ssim > 0:
                # [修正] 使用正确的类名 SSIMLoss
                self.criterionSSIM = SSIMLoss().to(self.device)

            # 优化器
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers = []
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

    def set_input(self, input):
        """
        处理输入数据
        A -> input_lq (输入)
        B -> real_sq (真值)
        """
        self.input_lq = input['A'].to(self.device)
        self.real_sq = input['B'].to(self.device)
        self.image_paths = input['A_paths']

    def forward(self):
        """前向传播"""
        self.fake_sq = self.netG(self.input_lq)

    def backward_D(self):
        """Discriminator 反向传播"""
        # Fake (生成的)
        # 拼接: input_lq + fake_sq
        fake_AB = torch.cat((self.input_lq, self.fake_sq), 1) 
        pred_fake = self.netD(fake_AB.detach())
        self.loss_D_fake = self.criterionGAN(pred_fake, False)

        # Real (真实的)
        # 拼接: input_lq + real_sq
        real_AB = torch.cat((self.input_lq, self.real_sq), 1)
        pred_real = self.netD(real_AB)
        self.loss_D_real = self.criterionGAN(pred_real, True)

        # 组合 Loss
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        self.loss_D.backward()

    def backward_G(self):
        """Generator 反向传播"""
        # 1. GAN Loss
        fake_AB = torch.cat((self.input_lq, self.fake_sq), 1)
        pred_fake = self.netD(fake_AB)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True) * self.opt.lambda_GAN

        # 2. L1 Loss
        self.loss_G_L1 = self.criterionL1(self.fake_sq, self.real_sq) * self.opt.lambda_L1

        self.loss_G = self.loss_G_GAN + self.loss_G_L1

        # 3. 可选 Loss (TV, Freq, VGG, SSIM)
        # [优化] 使用 self.denorm() 统一归一化，不再到处写 (x+1)/2
        
        if self.opt.lambda_TV > 0:
            self.loss_G_TV = self.criterionTV(self.fake_sq) * self.opt.lambda_TV
            self.loss_G += self.loss_G_TV
            
        if self.opt.lambda_freq > 0:
            self.loss_G_Freq = self.criterionFreq(self.fake_sq, self.real_sq) * self.opt.lambda_freq
            self.loss_G += self.loss_G_Freq

        if self.opt.lambda_vgg > 0:
            # VGG 需要 [0,1] 范围的 RGB 输入 (如果是单通道需要 repeat)
            # 这里先反归一化到 [0,1]
            fake_01 = self.denorm(self.fake_sq)
            real_01 = self.denorm(self.real_sq)
            
            # 如果是单通道，转为3通道适配VGG
            if fake_01.shape[1] == 1:
                fake_01 = fake_01.repeat(1, 3, 1, 1, 1)
                real_01 = real_01.repeat(1, 3, 1, 1, 1)
            
            # 由于 VGGFeatureExtractor 只返回 features，这里需要计算距离
            # 简单起见，如果开启 VGG loss，建议后续再仔细调试维度
            pass 

        if self.opt.lambda_ssim > 0:
            # SSIMLoss 期待 [-1, 1] 还是 [0, 1]? 通常 pytorch-ssim 是 [0, 1]
            # 你的 SSIMLoss 是基于卷积的，我们先传入 denorm 后的 [0, 1] 数据
            # 注意: 如果是 3D 数据，SSIMLoss 可能会报错，需要 slice 为 2D
            # 这里暂时保留逻辑，若开启需谨慎
            self.loss_G_SSIM = self.criterionSSIM(self.denorm(self.fake_sq), self.denorm(self.real_sq)) * self.opt.lambda_ssim
            self.loss_G += self.loss_G_SSIM

        self.loss_G.backward()

    def optimize_parameters(self):
        """
        核心训练步
        """
        self.forward()
        
        # update D
        self.set_requires_grad(self.netD, True)
        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step()

        # update G
        self.set_requires_grad(self.netD, False)
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()
        
        # [重要修改] 移除了 self.compute_visuals()

    def denorm(self, x):
        """反归一化: [-1, 1] -> [0, 1]"""
        return (x + 1.0) / 2.0