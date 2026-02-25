import torch
from .base_model import BaseModel
from . import networks
from .loss_helper import LossHelper

class AuganModel(BaseModel):
    """
    [Exp42 复刻版 - 修复测试模式报错] AUGAN Model
    修复点: 在初始化 loss_names 时增加 self.isTrain 判断，
           防止测试时因缺少 lambda_tv 等参数而报错。
    """
    def name(self):
        return 'AuganModel'

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.set_defaults(netG='res_unet_3d', norm='batch', dataset_mode='ultrasound')
        if is_train:
            parser.add_argument('--lambda_pixel', type=float, default=100.0, help='weight for L1 loss')
            parser.add_argument('--lambda_gan', type=float, default=1.0, help='weight for GAN loss')
            parser.add_argument('--lambda_tv', type=float, default=0.0, help='weight for TV loss')
            parser.add_argument('--lambda_ffl', type=float, default=0.0, help='weight for Frequency loss')
            parser.add_argument('--lambda_perceptual', type=float, default=0.0, help='weight for VGG perceptual loss')
            parser.add_argument('--lambda_ssim', type=float, default=0.0, help='weight for SSIM loss')
            parser.add_argument('--lambda_bg', type=float, default=0.0, help='weight for background suppression loss')
        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)
        self.isTrain = opt.isTrain
        
        # 兼容性逻辑
        if hasattr(opt, 'gan_mode'):
            use_sigmoid = (opt.gan_mode == 'vanilla')
        elif hasattr(opt, 'no_lsgan'):
            use_sigmoid = opt.no_lsgan
        else:
            use_sigmoid = False

        self.loss_names = []
        self.visual_names = ['img_lq', 'img_fake', 'img_sq']
        
        if self.isTrain:
            self.model_names = ['G', 'D']
            
            # [核心修复] 只有在训练模式下，才去访问这些 Loss 参数
            self.loss_names = ['G_GAN', 'G_L1', 'D_real', 'D_fake']
            if opt.lambda_tv > 0: self.loss_names.append('G_TV')
            if opt.lambda_ffl > 0: self.loss_names.append('G_Freq')
            if opt.lambda_perceptual > 0: self.loss_names.append('G_VGG')
            if opt.lambda_ssim > 0: self.loss_names.append('G_SSIM')
            if opt.lambda_bg > 0: self.loss_names.append('G_BG')
        else:
            self.model_names = ['G']

        # 1. Define G
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids,
                                      use_attention=opt.use_attention, attn_temp=opt.attn_temp, 
                                      use_dilation=opt.use_dilation, use_aspp=opt.use_aspp)

        # 2. Define D (仅训练)
        if self.isTrain:
            self.netD = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf, opt.netD,
                                          opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, opt.init_gain, self.gpu_ids, use_sn=getattr(opt, 'use_sn', False))

        # 3. 初始化 LossHelper (仅训练)
        if self.isTrain:
            self.loss_helper = LossHelper(opt, self.device)
            
            # 优化器
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers = []
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

    def set_input(self, input):
        # 优先寻找 'lq'/'sq'，找不到则回退到 'A'/'B'
        if 'lq' in input:
            self.input_lq = input['lq'].to(self.device)
        elif 'A' in input:
            self.input_lq = input['A'].to(self.device)
        else:
            # 测试时有些 Dataset 可能只返回 A/B
            pass

        if 'sq' in input:
            self.real_sq = input['sq'].to(self.device)
        elif 'hq' in input:
            self.real_sq = input['hq'].to(self.device)
        elif 'B' in input:
            self.real_sq = input['B'].to(self.device)
        else:
            pass

        if 'lq_path' in input:
            self.image_paths = input['lq_path']
        elif 'A_paths' in input:
            self.image_paths = input['A_paths']
        else:
            self.image_paths = [] 

    def compute_visuals(self):
        """
        将 3D 数据切片为 2D 图片，供 web_images 使用
        """
        if hasattr(self, 'input_lq') and self.input_lq is not None and self.input_lq.ndim == 5:
            mid = self.input_lq.size(2) // 2
            self.img_lq = self.input_lq[:, :, mid, :, :].detach()
            if hasattr(self, 'fake_sq') and self.fake_sq is not None:
                self.img_fake = self.fake_sq[:, :, mid, :, :].detach()
            if hasattr(self, 'real_sq') and self.real_sq is not None:
                self.img_sq = self.real_sq[:, :, mid, :, :].detach()
        else:
            # 兼容 2D 或其他情况
            pass

    def forward(self):
        self.fake_sq = self.netG(self.input_lq)

    def backward_D(self):
        # [调用 Helper] 计算 D Loss
        self.loss_D, loss_dict = self.loss_helper.compute_D_loss(
            self.netD, self.fake_sq, self.real_sq, self.input_lq
        )
        for k, v in loss_dict.items():
            setattr(self, 'loss_' + k, v)
            
        self.loss_D.backward()

    def backward_G(self):
        # [调用 Helper] 计算 G Loss
        self.loss_G, loss_dict = self.loss_helper.compute_G_loss(
            self.netD, self.fake_sq, self.real_sq, self.input_lq
        )
        for k, v in loss_dict.items():
            setattr(self, 'loss_' + k, v)

        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()
        self.set_requires_grad(self.netD, True)
        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step()

        self.set_requires_grad(self.netD, False)
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()
        
        self.compute_visuals()