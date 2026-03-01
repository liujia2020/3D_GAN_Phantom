import torch
import torch.nn as nn
import functools
import torch.nn.functional as F
from .blocks import ResnetBlock2D, ASPP2D, PixelAwareAttention, LocalAwareAttention

class ResUnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, padding_type='reflect', use_attention=False, attn_temp=5.0, use_dilation=False, use_aspp=False, upsample_mode='bilinear'):
        assert(n_blocks >= 0)
        super(ResUnetGenerator, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        is_3d = False # [核心修改] 彻底关闭 3D 模式

        unet_block = ResUnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True, is_3d=is_3d, use_attention=False, attn_temp=attn_temp, dilation=1, use_aspp=use_aspp, upsample_mode=upsample_mode)
        
        mid_dilation = 2 if use_dilation else 1
        
        for i in range(n_blocks - 5):
            unet_block = ResUnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout, is_3d=is_3d, use_attention=use_attention, attn_temp=attn_temp, dilation=mid_dilation, upsample_mode=upsample_mode)
        
        unet_block = ResUnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, is_3d=is_3d, use_attention=use_attention, attn_temp=attn_temp, dilation=1, upsample_mode=upsample_mode)
        unet_block = ResUnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer, is_3d=is_3d, use_attention=use_attention, attn_temp=attn_temp, dilation=1, upsample_mode=upsample_mode)
        unet_block = ResUnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer, is_3d=is_3d, use_attention=use_attention, attn_temp=attn_temp, dilation=1, upsample_mode=upsample_mode)
        
        self.model = ResUnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer, is_3d=is_3d, use_attention=use_attention, attn_temp=attn_temp, dilation=1, upsample_mode=upsample_mode)

    def forward(self, input):
        return self.model(input)

class ResUnetSkipConnectionBlock(nn.Module):
    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False, is_3d=False, use_attention=False, attn_temp=1.0, dilation=1, use_aspp=False, upsample_mode='bilinear'): 
        super(ResUnetSkipConnectionBlock, self).__init__()
        self.outermost = outermostc
        self.innermost = innermost
        self.use_attention = use_attention
        self.is_3d = False

        if input_nc is None: input_nc = outer_nc
        Conv = nn.Conv2d
        use_bias = norm_layer == nn.InstanceNorm2d

        downconv = Conv(input_nc, inner_nc, kernel_size=4, stride=2, padding=1, bias=use_bias, padding_mode='reflect')
        downrelu = nn.LeakyReLU(0.2, False)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(False)
        upnorm = norm_layer(outer_nc)
        
        enc_res_block = ResnetBlock2D(inner_nc, norm_layer, use_dropout, use_bias, dilation=dilation)
        dec_res_block = ResnetBlock2D(outer_nc, norm_layer, use_dropout, use_bias, dilation=dilation)

        mode_to_use = 'nearest' if upsample_mode == 'nearest' else 'bilinear'
        kwargs = {} if mode_to_use == 'nearest' else {'align_corners': False}

        if outermost:
            up_layer = nn.Sequential(
                nn.Upsample(scale_factor=2, mode=mode_to_use, **kwargs),
                Conv(inner_nc * 2, outer_nc, kernel_size=3, stride=1, padding=1, bias=True, padding_mode='reflect')
            )
            model = [downconv, enc_res_block] + [submodule] + [uprelu, up_layer, nn.Tanh()] 
            
        elif innermost:
            up_layer = nn.Sequential(
                nn.Upsample(scale_factor=2, mode=mode_to_use, **kwargs),
                Conv(inner_nc, outer_nc, kernel_size=3, stride=1, padding=1, bias=use_bias, padding_mode='reflect')
            )
            if use_aspp:
                aspp_layer = ASPP2D(inner_nc, inner_nc, norm_layer=norm_layer)
                model = [downrelu, downconv, aspp_layer, up_layer, upnorm]
            else:
                model = [downrelu, downconv, enc_res_block] + [uprelu, up_layer, upnorm, dec_res_block] 
            
        else:
            up_layer = nn.Sequential(
                nn.Upsample(scale_factor=2, mode=mode_to_use, **kwargs),
                Conv(inner_nc * 2, outer_nc, kernel_size=3, stride=1, padding=1, bias=use_bias, padding_mode='reflect')
            )
            down = [downrelu, downconv, downnorm, enc_res_block]
            up = [uprelu, up_layer, upnorm, dec_res_block]
            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

        if self.use_attention and not self.innermost:
            self.pa = PixelAwareAttention(input_nc, is_3d=False, temperature=attn_temp)
            self.laa = LocalAwareAttention(F_g=outer_nc, F_l=input_nc, F_int=max(input_nc//2, 1), is_3d=False)

    def forward(self, x):
        if self.outermost: return self.model(x)
        else:
            decoder_feature = self.model(x) 
            skip_feature = x
            if self.use_attention and not self.innermost:
                skip_feature = self.pa(skip_feature)
                skip_feature = self.laa(g=decoder_feature, x=skip_feature)
            return torch.cat([skip_feature, decoder_feature], 1)