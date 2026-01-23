import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.optim import lr_scheduler

# 1. 导入 Generator
from .generator import UnetGenerator3D

# 2. 导入 Discriminator
from .discriminator import NLayerDiscriminator3D

# ==============================================================================
# [补丁] PixelDiscriminator3D (保留着防止报错，但这次不用它)
# ==============================================================================
class PixelDiscriminator3D(nn.Module):
    def __init__(self, input_nc, ndf=64, norm_layer=nn.BatchNorm3d, use_sigmoid=False):
        super(PixelDiscriminator3D, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm3d
        else:
            use_bias = norm_layer == nn.InstanceNorm3d

        self.net = [
            nn.Conv3d(input_nc, ndf, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, True),
            nn.Conv3d(ndf, ndf * 2, kernel_size=1, stride=1, padding=0, bias=use_bias),
            norm_layer(ndf * 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv3d(ndf * 2, 1, kernel_size=1, stride=1, padding=0, bias=use_bias)
        ]
        if use_sigmoid:
            self.net.append(nn.Sigmoid())
        self.net = nn.Sequential(*self.net)

    def forward(self, input):
        return self.net(input)

# ==============================================================================
# define_G
# ==============================================================================
def define_G(input_nc, output_nc, ngf, netG, norm='batch', use_dropout=False, init_type='normal', init_gain=0.02, gpu_ids=[]):
    net = None
    norm_layer = get_norm_layer(norm_type=norm)

    if netG == 'unet_3d':
        net = UnetGenerator3D(input_nc, output_nc, num_downs=4, ngf=ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % netG)

    return init_net(net, init_type, init_gain, gpu_ids)

# ==============================================================================
# [关键修改] define_D
# ==============================================================================
def define_D(input_nc, ndf, netD, n_layers_D=3, norm='batch', use_sigmoid=False, init_type='normal', init_gain=0.02, gpu_ids=[]):
    net = None
    norm_layer = get_norm_layer(norm_type=norm)

    # [救档逻辑]
    # 虽然参数叫 'pixel'，但你的存档里其实是 NLayerDiscriminator (PatchGAN)。
    # 所以这里强制指向 NLayerDiscriminator3D，并保持 n_layers=3 (默认值)
    if netD == 'pixel':
        # print("⚠️ 检测到 'pixel' 参数，但为了匹配旧权重，强制使用 NLayerDiscriminator3D (PatchGAN)...")
        net = NLayerDiscriminator3D(input_nc, ndf, n_layers=3, norm_layer=norm_layer) # 强制 n_layers=3 以匹配权重
        
    elif netD == 'n_layers':
        net = NLayerDiscriminator3D(input_nc, ndf, n_layers=n_layers_D, norm_layer=norm_layer)
    else:
        # 默认 fallback
        net = NLayerDiscriminator3D(input_nc, ndf, n_layers=3, norm_layer=norm_layer)

    return init_net(net, init_type, init_gain, gpu_ids)

# # ==============================================================================
# # 辅助函数
# # ==============================================================================
# def get_norm_layer(norm_type='instance'):
#     if norm_type == 'batch':
#         norm_layer = functools.partial(nn.BatchNorm3d, affine=True, track_running_stats=True)
#     elif norm_type == 'instance':
#         norm_layer = functools.partial(nn.InstanceNorm3d, affine=False, track_running_stats=False)
#     elif norm_type == 'none':
#         def norm_layer(x): return nn.Identity()
#     else:
#         raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
#     return norm_layer
def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm3d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm3d, affine=False, track_running_stats=False)
    elif norm_type == 'group':
        # [新增] Group Norm: 最适合 Batch Size 小且数据稀疏的场景
        # num_groups=32 是经典默认值，你的 ngf=64，完全可以被整除
        norm_layer = functools.partial(nn.GroupNorm, 32)
    elif norm_type == 'none':
        def norm_layer(x): return nn.Identity()
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def init_weights(net, init_type='normal', init_gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm3d') != -1:
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)
    # print('initialize network with %s' % init_type)
    net.apply(init_func)

def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
    init_weights(net, init_type, init_gain=init_gain)
    return net

class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0):
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCEWithLogitsLoss()
    def get_target_tensor(self, prediction, target_is_real):
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)
    def __call__(self, prediction, target_is_real):
        target_tensor = self.get_target_tensor(prediction, target_is_real)
        loss = self.loss(prediction, target_tensor)
        return loss