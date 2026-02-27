import torch
import torch.nn as nn
import functools
import torch.nn.functional as F
from .blocks import ResnetBlock3D, ASPP3D, PixelAwareAttention, LocalAwareAttention

class ResUnetGenerator(nn.Module):
    # [修改] 增加 upsample_mode='trilinear'
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm3d, use_dropout=False, n_blocks=6, padding_type='reflect', use_attention=False, attn_temp=5.0, use_dilation=False, use_aspp=False, upsample_mode='trilinear'):
        assert(n_blocks >= 0)
        super(ResUnetGenerator, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        is_3d = True 

        # [修改] 所有 Block 初始化时都要带上 upsample_mode=upsample_mode
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
    # [修改] 增加 upsample_mode='trilinear'
    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm3d, use_dropout=False, is_3d=True, use_attention=False, attn_temp=1.0, dilation=1, use_aspp=False, upsample_mode='trilinear'): 
        super(ResUnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        self.innermost = innermost
        self.use_attention = use_attention
        self.is_3d = is_3d

        if input_nc is None: input_nc = outer_nc
        Conv = nn.Conv3d if is_3d else nn.Conv2d
        use_bias = norm_layer == nn.InstanceNorm3d

        downconv = Conv(input_nc, inner_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, False)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(False)
        upnorm = norm_layer(outer_nc)
        
        enc_res_block = ResnetBlock3D(inner_nc, norm_layer, use_dropout, use_bias, dilation=dilation)
        dec_res_block = ResnetBlock3D(outer_nc, norm_layer, use_dropout, use_bias, dilation=dilation)

        # [新增] 核心适配逻辑：安全处理 nearest 和 align_corners 的冲突
        mode_to_use = upsample_mode if is_3d else ('nearest' if upsample_mode == 'nearest' else 'bilinear')
        kwargs = {} if mode_to_use == 'nearest' else {'align_corners': False}

        if outermost:
            up_layer = nn.Sequential(
                # [修改] 替换写死的 trilinear
                nn.Upsample(scale_factor=2, mode=mode_to_use, **kwargs),
                Conv(inner_nc * 2, outer_nc, kernel_size=3, stride=1, padding=1, bias=True)
            )
            model = [downconv, enc_res_block] + [submodule] + [uprelu, up_layer, nn.Tanh()] 
            
        elif innermost:
            up_layer = nn.Sequential(
                # [修改] 替换写死的 trilinear
                nn.Upsample(scale_factor=2, mode=mode_to_use, **kwargs),
                Conv(inner_nc, outer_nc, kernel_size=3, stride=1, padding=1, bias=use_bias)
            )
            if use_aspp:
                aspp_layer = ASPP3D(inner_nc, inner_nc, norm_layer=norm_layer)
                model = [downrelu, downconv, aspp_layer, up_layer, upnorm]
            else:
                model = [downrelu, downconv, enc_res_block] + [uprelu, up_layer, upnorm, dec_res_block] 
            
        else:
            up_layer = nn.Sequential(
                # [修改] 替换写死的 trilinear
                nn.Upsample(scale_factor=2, mode=mode_to_use, **kwargs),
                Conv(inner_nc * 2, outer_nc, kernel_size=3, stride=1, padding=1, bias=use_bias)
            )
            down = [downrelu, downconv, downnorm, enc_res_block]
            up = [uprelu, up_layer, upnorm, dec_res_block]
            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

        if self.use_attention and not self.innermost:
            self.pa = PixelAwareAttention(input_nc, is_3d=is_3d, temperature=attn_temp)
            self.laa = LocalAwareAttention(F_g=outer_nc, F_l=input_nc, F_int=max(input_nc//2, 1), is_3d=is_3d)

    def forward(self, x):
        if self.outermost: return self.model(x)
        else:
            decoder_feature = self.model(x) 
            skip_feature = x
            if self.use_attention and not self.innermost:
                skip_feature = self.pa(skip_feature)
                skip_feature = self.laa(g=decoder_feature, x=skip_feature)
            return torch.cat([skip_feature, decoder_feature], 1)

# import torch
# import torch.nn as nn
# import functools
# import torch.nn.functional as F
# # [新增] 从 blocks.py 引入基础组件
# from .blocks import ResnetBlock3D, ASPP3D, PixelAwareAttention, LocalAwareAttention

# # ==========================================================
# # [核心类] ResUnetGenerator
# # ==========================================================

# class ResUnetGenerator(nn.Module):
#     """
#     [Exp 30 核心]: ResUnet Generator (CBAM Enabled via ResnetBlock3D)
#     """
#     def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm3d, use_dropout=False, n_blocks=6, padding_type='reflect', use_attention=False, attn_temp=5.0, use_dilation=False, use_aspp=False):
#         assert(n_blocks >= 0)
#         super(ResUnetGenerator, self).__init__()
#         self.input_nc = input_nc
#         self.output_nc = output_nc
#         self.ngf = ngf
        
#         # [关键修复]: 在内部显式定义 is_3d，不再依赖外部传参
#         is_3d = True 

#         # 1. Innermost (最内层): 强制 dilation=1，【开启 ASPP】
#         unet_block = ResUnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True, is_3d=is_3d, use_attention=False, attn_temp=attn_temp, dilation=1, use_aspp=use_aspp)
        
#         # 2. Intermediate blocks (深层中间层)
#         mid_dilation = 2 if use_dilation else 1
        
#         for i in range(n_blocks - 5):
#             unet_block = ResUnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout, is_3d=is_3d, use_attention=use_attention, attn_temp=attn_temp, dilation=mid_dilation)
        
#         # 3. Up-sampling blocks (浅层)
#         unet_block = ResUnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, is_3d=is_3d, use_attention=use_attention, attn_temp=attn_temp, dilation=1)
#         unet_block = ResUnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer, is_3d=is_3d, use_attention=use_attention, attn_temp=attn_temp, dilation=1)
#         unet_block = ResUnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer, is_3d=is_3d, use_attention=use_attention, attn_temp=attn_temp, dilation=1)
        
#         # Outermost block
#         self.model = ResUnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer, is_3d=is_3d, use_attention=use_attention, attn_temp=attn_temp, dilation=1)

#     def forward(self, input):
#         return self.model(input)


# class ResUnetSkipConnectionBlock(nn.Module):
#     """
#     [Exp 30 核心]: ResUnet Block
#     """
#     def __init__(self, outer_nc, inner_nc, input_nc=None,
#                  submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm3d, use_dropout=False, is_3d=True, use_attention=False, attn_temp=1.0, dilation=1, use_aspp=False): 
#         super(ResUnetSkipConnectionBlock, self).__init__()
#         self.outermost = outermost
#         self.innermost = innermost
#         self.use_attention = use_attention
#         self.is_3d = is_3d

#         if input_nc is None: input_nc = outer_nc
        
#         Conv = nn.Conv3d if is_3d else nn.Conv2d
#         use_bias = norm_layer == nn.InstanceNorm3d

#         # Downsample
#         downconv = Conv(input_nc, inner_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
#         downrelu = nn.LeakyReLU(0.2, False)
#         downnorm = norm_layer(inner_nc)
#         uprelu = nn.ReLU(False)
#         upnorm = norm_layer(outer_nc)
        
#         # ResBlock (New: Now contains CBAM implicitly)
#         # [修改] 直接使用引入的 ResnetBlock3D
#         enc_res_block = ResnetBlock3D(inner_nc, norm_layer, use_dropout, use_bias, dilation=dilation)
#         dec_res_block = ResnetBlock3D(outer_nc, norm_layer, use_dropout, use_bias, dilation=dilation)

#         # Upsample + Conv
#         if outermost:
#             up_layer = nn.Sequential(
#                 nn.Upsample(scale_factor=2, mode='trilinear' if is_3d else 'bilinear', align_corners=False),
#                 Conv(inner_nc * 2, outer_nc, kernel_size=3, stride=1, padding=1, bias=True)
#             )
#             down = [downconv, enc_res_block] 
#             up = [uprelu, up_layer, nn.Tanh()] 
#             model = down + [submodule] + up
            
#         elif innermost:
#             up_layer = nn.Sequential(
#                 nn.Upsample(scale_factor=2, mode='trilinear' if is_3d else 'bilinear', align_corners=False),
#                 Conv(inner_nc, outer_nc, kernel_size=3, stride=1, padding=1, bias=use_bias)
#             )
            
#             # [ASPP Logic] - 使用引入的 ASPP3D
#             if use_aspp:
#                 aspp_layer = ASPP3D(inner_nc, inner_nc, norm_layer=norm_layer)
#                 model = [downrelu, downconv, aspp_layer, up_layer, upnorm]
#             else:
#                 down = [downrelu, downconv, enc_res_block] 
#                 up = [uprelu, up_layer, upnorm, dec_res_block] 
#                 model = down + up
            
#         else:
#             up_layer = nn.Sequential(
#                 nn.Upsample(scale_factor=2, mode='trilinear' if is_3d else 'bilinear', align_corners=False),
#                 Conv(inner_nc * 2, outer_nc, kernel_size=3, stride=1, padding=1, bias=use_bias)
#             )
#             down = [downrelu, downconv, downnorm, enc_res_block]
#             up = [uprelu, up_layer, upnorm, dec_res_block]
#             if use_dropout:
#                 model = down + [submodule] + up + [nn.Dropout(0.5)]
#             else:
#                 model = down + [submodule] + up

#         self.model = nn.Sequential(*model)

#         if self.use_attention and not self.innermost:
#             # [修改] 使用引入的 Attention 模块
#             self.pa = PixelAwareAttention(input_nc, is_3d=is_3d, temperature=attn_temp)
#             self.laa = LocalAwareAttention(F_g=outer_nc, F_l=input_nc, F_int=max(input_nc//2, 1), is_3d=is_3d)

#     def forward(self, x):
#         if self.outermost: return self.model(x)
#         else:
#             decoder_feature = self.model(x) 
#             skip_feature = x
#             if self.use_attention and not self.innermost:
#                 skip_feature = self.pa(skip_feature)
#                 skip_feature = self.laa(g=decoder_feature, x=skip_feature)
#             return torch.cat([skip_feature, decoder_feature], 1)
