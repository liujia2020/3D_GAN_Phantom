import torch
from .base_model import BaseModel
from . import networks
from .losses import FrequencyLoss
from utils.ssim_loss import SSIMLoss

class AuganModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.set_defaults(norm='batch', netG='res_unet_3d', dataset_mode='ultrasound')
        if is_train:
            parser.add_argument('--lambda_pixel', type=float, default=100.0, help='weight for L1 loss')
            parser.add_argument('--lambda_gan', type=float, default=1.0, help='weight for GAN loss')
            parser.add_argument('--lambda_ssim', type=float, default=10.0, help='weight for SSIM loss')
            parser.add_argument('--lambda_ffl', type=float, default=10.0, help='weight for FFL loss')
        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)
        self.loss_names = ['G_GAN', 'G_L1', 'D_real', 'D_fake', 'G_SSIM', 'G_FFL']
        self.visual_names = ['real_LQ_mid', 'fake_HQ', 'real_SQ']
        
        if self.isTrain:
            self.model_names = ['G', 'D']
        else:
            self.model_names = ['G']

        # 初始化 2D 网络
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids,
                                      use_attention=opt.use_attention, attn_temp=opt.attn_temp, 
                                      use_dilation=opt.use_dilation, use_aspp=opt.use_aspp, 
                                      upsample_mode=opt.upsample_mode)

        if self.isTrain:
            # 判别器的输入通道 = LQ的3通道 + HQ的1通道 = 4通道
            # 修复 no_lsgan 报错：直接写死 use_sigmoid=False
            self.netD = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf, opt.netD,
                                          opt.n_layers_D, opt.norm, False, opt.init_type, opt.init_gain, self.gpu_ids, use_sn=opt.use_sn)

            # 修复 no_lsgan 报错：直接写死 use_lsgan=True
            self.criterionGAN = networks.GANLoss(use_lsgan=True).to(self.device)
            self.criterionL1 = torch.nn.L1Loss()
            self.criterionSSIM = SSIMLoss(window_size=11, val_range=2.0).to(self.device)
            # 修复 FrequencyLoss 名称不匹配
            self.criterionFFL = FrequencyLoss().to(self.device)
            
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

    def set_input(self, input):
        # real_LQ 是 3 个通道 (z-1, z, z+1)
        self.real_LQ = input['lq'].to(self.device)
        # real_SQ 是 1 个通道 (z)
        self.real_SQ = input['sq'].to(self.device)
        self.image_paths = input['lq_path']

        # 可视化时，提取夹心饼干中间那一层 (索引 1) 供前端查看
        if self.real_LQ.shape[1] == 3:
            self.real_LQ_mid = self.real_LQ[:, 1:2, :, :]
        else:
            self.real_LQ_mid = self.real_LQ

    def forward(self):
        self.fake_HQ = self.netG(self.real_LQ) # 吐出 1 个通道

    def backward_D(self):
        # 拼接策略：3 通道 LQ + 1 通道假/真图 = 4 通道特征喂给判别器
        fake_AB = torch.cat((self.real_LQ, self.fake_HQ), 1)
        pred_fake = self.netD(fake_AB.detach())
        self.loss_D_fake = self.criterionGAN(pred_fake, False)

        real_AB = torch.cat((self.real_LQ, self.real_SQ), 1)
        pred_real = self.netD(real_AB)
        self.loss_D_real = self.criterionGAN(pred_real, True)

        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        self.loss_D.backward()

    def backward_G(self):
        fake_AB = torch.cat((self.real_LQ, self.fake_HQ), 1)
        pred_fake = self.netD(fake_AB)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True) * self.opt.lambda_gan

        # L1, SSIM, FFL 只在 1 通道的假图和真图之间计算，公平公正
        self.loss_G_L1 = self.criterionL1(self.fake_HQ, self.real_SQ) * self.opt.lambda_pixel
        self.loss_G_SSIM = self.criterionSSIM(self.fake_HQ, self.real_SQ) * self.opt.lambda_ssim
        self.loss_G_FFL = self.criterionFFL(self.fake_HQ, self.real_SQ) * self.opt.lambda_ffl

        self.loss_G = self.loss_G_GAN + self.loss_G_L1 + self.loss_G_SSIM + self.loss_G_FFL
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