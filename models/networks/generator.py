import torch
import torch.nn as nn
import functools
import torch.nn.functional as F

# ==========================================================
# Part 0: CBAM 核心组件 (Exp 52 新增)
# ==========================================================

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)
           
        # 使用 1x1x1 卷积代替全连接层，减少参数量
        self.fc = nn.Sequential(
            nn.Conv3d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv3d(in_planes // ratio, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv3d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 在通道维度上做平均和最大池化
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class CBAM3D(nn.Module):
    def __init__(self, planes, ratio=16, kernel_size=7):
        super(CBAM3D, self).__init__()
        self.ca = ChannelAttention(planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        out = x * self.ca(x)
        result = out * self.sa(out)
        return result

# ==========================================================
# Part 1: 核心组件 (Exp 30 & Exp 52)
# ==========================================================

class ResnetBlock3D(nn.Module):
    """
    [Exp 52 修改]: 集成了 CBAM 模块的 3D 残差块
    """
    def __init__(self, dim, norm_layer, use_dropout, use_bias, dilation=1):
        super(ResnetBlock3D, self).__init__()
        self.conv_block = self.build_conv_block(dim, norm_layer, use_dropout, use_bias, dilation)
        self.relu = nn.ReLU(False)
        
        # [新增] 初始化 CBAM
        self.cbam = CBAM3D(dim, ratio=16, kernel_size=7)

    def build_conv_block(self, dim, norm_layer, use_dropout, use_bias, dilation):
        conv_block = []
        p = dilation # Padding = Dilation 以保持尺寸不变
        
        # First Conv Layer
        conv_block += [nn.Conv3d(dim, dim, kernel_size=3, padding=p, dilation=dilation, bias=use_bias),
                       norm_layer(dim),
                       nn.ReLU(False)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        # Second Conv Layer
        conv_block += [nn.Conv3d(dim, dim, kernel_size=3, padding=p, dilation=dilation, bias=use_bias),
                       norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        residual = self.conv_block(x)
        
        # [Exp 52]: 在残差路径上应用 CBAM
        residual = self.cbam(residual)
        
        out = x + residual
        return self.relu(out)


class PixelAwareAttention(nn.Module):
    """
    [Exp 30 核心]: 支持 temperature 锐化的 Pixel Attention
    """
    def __init__(self, in_channels, is_3d=True, temperature=1.0):
        super(PixelAwareAttention, self).__init__()
        Conv = nn.Conv3d if is_3d else nn.Conv2d
        self.conv1 = Conv(in_channels, 1, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.temperature = temperature

    def forward(self, x):
        attn_map = self.sigmoid(self.conv1(x) * self.temperature)
        return x * attn_map


class LocalAwareAttention(nn.Module):
    """Local Aware Attention (保持不变)"""
    def __init__(self, F_g, F_l, F_int, is_3d=True):
        super(LocalAwareAttention, self).__init__()
        Conv = nn.Conv3d if is_3d else nn.Conv2d
        Bn = nn.BatchNorm3d if is_3d else nn.BatchNorm2d
        
        self.W_g = nn.Sequential(
            Conv(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            Bn(F_int)
        )
        self.W_x = nn.Sequential(
            Conv(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            Bn(F_int)
        )
        self.psi = nn.Sequential(
            Conv(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            Bn(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=False)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        if g1.shape[2:] != x1.shape[2:]:
            g1 = torch.nn.functional.interpolate(g1, size=x1.shape[2:], mode='trilinear', align_corners=False)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi

class ASPP3D(nn.Module):
    """
    [Exp 51 修复版]: 降低膨胀率，防止在小特征图上出现空洞
    Rates: [1, 2, 3] -> 安全适配 8x8 特征图
    """
    def __init__(self, in_channels, out_channels, norm_layer=nn.BatchNorm3d):
        super(ASPP3D, self).__init__()
        
        # Branch 1: Dilation 1 (1x1 conv)
        self.branch1 = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False),
            norm_layer(out_channels),
            nn.LeakyReLU(0.2, True)
        )
        
        # Branch 2: Dilation 2
        self.branch2 = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=2, dilation=2, bias=False),
            norm_layer(out_channels),
            nn.LeakyReLU(0.2, True)
        )
        
        # Branch 3: Dilation 3
        self.branch3 = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=3, dilation=3, bias=False),
            norm_layer(out_channels),
            nn.LeakyReLU(0.2, True)
        )
        
        # Branch 5: Global Pooling
        self.branch5_conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False),
            norm_layer(out_channels),
            nn.LeakyReLU(0.2, True)
        )

        # Fusion
        self.conv_cat = nn.Sequential(
            nn.Conv3d(out_channels * 4, out_channels, kernel_size=1, bias=False),
            norm_layer(out_channels),
            nn.LeakyReLU(0.2, True)
        )

    def forward(self, x):
        [b, c, d, h, w] = x.size()
        
        conv1x1 = self.branch1(x)
        conv3x3_d2 = self.branch2(x)
        conv3x3_d3 = self.branch3(x)
        
        # Global Pooling
        global_feature = F.adaptive_avg_pool3d(x, (1, 1, 1))
        global_feature = self.branch5_conv(global_feature)
        global_feature = F.interpolate(global_feature, size=(d, h, w), mode='trilinear', align_corners=True)
        
        # Concat
        feature_cat = torch.cat([conv1x1, conv3x3_d2, conv3x3_d3, global_feature], dim=1)
        
        result = self.conv_cat(feature_cat)
        return result   

class ResUnetGenerator(nn.Module):
    """
    [Exp 30 核心]: ResUnet Generator (CBAM Enabled via ResnetBlock3D)
    """
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm3d, use_dropout=False, n_blocks=6, padding_type='reflect', use_attention=False, attn_temp=5.0, use_dilation=False, use_aspp=False):
        assert(n_blocks >= 0)
        super(ResUnetGenerator, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        
        # [关键修复]: 在内部显式定义 is_3d，不再依赖外部传参
        is_3d = True 

        # 1. Innermost (最内层): 强制 dilation=1，【开启 ASPP】
        unet_block = ResUnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True, is_3d=is_3d, use_attention=False, attn_temp=attn_temp, dilation=1, use_aspp=use_aspp)
        
        # 2. Intermediate blocks (深层中间层)
        mid_dilation = 2 if use_dilation else 1
        
        for i in range(n_blocks - 5):
            unet_block = ResUnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout, is_3d=is_3d, use_attention=use_attention, attn_temp=attn_temp, dilation=mid_dilation)
        
        # 3. Up-sampling blocks (浅层)
        unet_block = ResUnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, is_3d=is_3d, use_attention=use_attention, attn_temp=attn_temp, dilation=1)
        unet_block = ResUnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer, is_3d=is_3d, use_attention=use_attention, attn_temp=attn_temp, dilation=1)
        unet_block = ResUnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer, is_3d=is_3d, use_attention=use_attention, attn_temp=attn_temp, dilation=1)
        
        # Outermost block
        self.model = ResUnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer, is_3d=is_3d, use_attention=use_attention, attn_temp=attn_temp, dilation=1)

    def forward(self, input):
        return self.model(input)


class ResUnetSkipConnectionBlock(nn.Module):
    """
    [Exp 30 核心]: ResUnet Block
    """
    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm3d, use_dropout=False, is_3d=True, use_attention=False, attn_temp=1.0, dilation=1, use_aspp=False): 
        super(ResUnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        self.innermost = innermost
        self.use_attention = use_attention
        self.is_3d = is_3d

        if input_nc is None: input_nc = outer_nc
        
        Conv = nn.Conv3d if is_3d else nn.Conv2d
        use_bias = norm_layer == nn.InstanceNorm3d

        # Downsample
        downconv = Conv(input_nc, inner_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, False)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(False)
        upnorm = norm_layer(outer_nc)
        
        # ResBlock (New: Now contains CBAM implicitly)
        enc_res_block = ResnetBlock3D(inner_nc, norm_layer, use_dropout, use_bias, dilation=dilation)
        dec_res_block = ResnetBlock3D(outer_nc, norm_layer, use_dropout, use_bias, dilation=dilation)

        # =================================================================
        # [修改] 使用 Upsample + Conv 代替 ConvTranspose3d 以消除棋盘格
        # =================================================================
        if outermost:
            up_layer = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='trilinear' if is_3d else 'bilinear', align_corners=False),
                Conv(inner_nc * 2, outer_nc, kernel_size=3, stride=1, padding=1, bias=True)
            )
            down = [downconv, enc_res_block] 
            up = [uprelu, up_layer, nn.Tanh()] 
            model = down + [submodule] + up
            
        elif innermost:
            up_layer = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='trilinear' if is_3d else 'bilinear', align_corners=False),
                Conv(inner_nc, outer_nc, kernel_size=3, stride=1, padding=1, bias=use_bias)
            )
            
            # [ASPP Logic]
            if use_aspp:
                aspp_layer = ASPP3D(inner_nc, inner_nc, norm_layer=norm_layer)
                model = [downrelu, downconv, aspp_layer, up_layer, upnorm]
            else:
                down = [downrelu, downconv, enc_res_block] 
                up = [uprelu, up_layer, upnorm, dec_res_block] 
                model = down + up
            
        else:
            up_layer = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='trilinear' if is_3d else 'bilinear', align_corners=False),
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

# ==========================================================
# Part 2: 遗留兼容组件 (Standard Unet) - [必须保留以防报错]
# ==========================================================

class UnetGenerator(nn.Module):
    """原始 Standard U-Net Generator (用于兼容旧代码引用)"""
    def __init__(self, input_nc, output_nc, num_downs, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, is_3d=False, use_attention=False):
        super(UnetGenerator, self).__init__()
        # 构造 unet 结构
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True, is_3d=is_3d, use_attention=False)
        for i in range(num_downs - 5): 
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout, is_3d=is_3d, use_attention=use_attention)
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, is_3d=is_3d, use_attention=use_attention)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer, is_3d=is_3d, use_attention=use_attention)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer, is_3d=is_3d, use_attention=use_attention)
        self.model = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer, is_3d=is_3d, use_attention=use_attention)

    def forward(self, input):
        return self.model(input)

class UnetSkipConnectionBlock(nn.Module):
    """
    [修改]: 原始 Unet Block 也改为 Upsample 以防止潜在的混用伪影
    """
    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False, is_3d=False, use_attention=False):
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        self.innermost = innermost
        self.use_attention = use_attention
        self.is_3d = is_3d
        if input_nc is None: input_nc = outer_nc
        
        Conv = nn.Conv3d if is_3d else nn.Conv2d
        # 删除 ConvTrans
        # ConvTrans = nn.ConvTranspose3d if is_3d else nn.ConvTranspose2d
        
        downconv = Conv(input_nc, inner_nc, kernel_size=4, stride=2, padding=1, bias=False)
        downrelu = nn.LeakyReLU(0.2, False)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(False)
        upnorm = norm_layer(outer_nc)

        # [修改] 使用 Upsample + Conv 代替 ConvTranspose
        if outermost:
            up_layer = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='trilinear' if is_3d else 'bilinear', align_corners=False),
                Conv(inner_nc * 2, outer_nc, kernel_size=3, stride=1, padding=1, bias=True)
            )
            down = [downconv]
            up = [uprelu, up_layer, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            up_layer = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='trilinear' if is_3d else 'bilinear', align_corners=False),
                Conv(inner_nc, outer_nc, kernel_size=3, stride=1, padding=1, bias=False)
            )
            down = [downrelu, downconv]
            up = [uprelu, up_layer, upnorm]
            model = down + up
        else:
            up_layer = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='trilinear' if is_3d else 'bilinear', align_corners=False),
                Conv(inner_nc * 2, outer_nc, kernel_size=3, stride=1, padding=1, bias=False)
            )
            down = [downrelu, downconv, downnorm]
            up = [uprelu, up_layer, upnorm]
            if use_dropout: model = down + [submodule] + up + [nn.Dropout(0.5)]
            else: model = down + [submodule] + up
        self.model = nn.Sequential(*model)
        
        if self.use_attention and not self.innermost:
            self.pa = PixelAwareAttention(input_nc, is_3d=is_3d)
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

# 兼容性 Wrapper (用于 netG=='unet_3d' 时)
class UnetGenerator3D(UnetGenerator):
    def __init__(self, input_nc, output_nc, num_downs, ngf=64, norm_layer=nn.BatchNorm3d, use_dropout=False, use_attention=False):
        super(UnetGenerator3D, self).__init__(input_nc, output_nc, num_downs, ngf, norm_layer, use_dropout, is_3d=True, use_attention=use_attention)

# import torch
# import torch.nn as nn
# import functools
# import torch.nn.functional as F

# # ==========================================================
# # Part 0: CBAM 核心组件 (Exp 52 新增)
# # ==========================================================

# class ChannelAttention(nn.Module):
#     def __init__(self, in_planes, ratio=16):
#         super(ChannelAttention, self).__init__()
#         self.avg_pool = nn.AdaptiveAvgPool3d(1)
#         self.max_pool = nn.AdaptiveMaxPool3d(1)
           
#         # 使用 1x1x1 卷积代替全连接层，减少参数量
#         self.fc = nn.Sequential(
#             nn.Conv3d(in_planes, in_planes // ratio, 1, bias=False),
#             nn.ReLU(),
#             nn.Conv3d(in_planes // ratio, in_planes, 1, bias=False)
#         )
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x):
#         avg_out = self.fc(self.avg_pool(x))
#         max_out = self.fc(self.max_pool(x))
#         out = avg_out + max_out
#         return self.sigmoid(out)

# class SpatialAttention(nn.Module):
#     def __init__(self, kernel_size=7):
#         super(SpatialAttention, self).__init__()
#         assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
#         padding = 3 if kernel_size == 7 else 1
#         self.conv1 = nn.Conv3d(2, 1, kernel_size, padding=padding, bias=False)
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x):
#         # 在通道维度上做平均和最大池化
#         avg_out = torch.mean(x, dim=1, keepdim=True)
#         max_out, _ = torch.max(x, dim=1, keepdim=True)
#         x = torch.cat([avg_out, max_out], dim=1)
#         x = self.conv1(x)
#         return self.sigmoid(x)

# class CBAM3D(nn.Module):
#     def __init__(self, planes, ratio=16, kernel_size=7):
#         super(CBAM3D, self).__init__()
#         self.ca = ChannelAttention(planes, ratio)
#         self.sa = SpatialAttention(kernel_size)

#     def forward(self, x):
#         out = x * self.ca(x)
#         result = out * self.sa(out)
#         return result

# # ==========================================================
# # Part 1: 核心组件 (Exp 30 & Exp 52)
# # ==========================================================

# class ResnetBlock3D(nn.Module):
#     """
#     [Exp 52 修改]: 集成了 CBAM 模块的 3D 残差块
#     """
#     def __init__(self, dim, norm_layer, use_dropout, use_bias, dilation=1):
#         super(ResnetBlock3D, self).__init__()
#         self.conv_block = self.build_conv_block(dim, norm_layer, use_dropout, use_bias, dilation)
#         self.relu = nn.ReLU(False)
        
#         # [新增] 初始化 CBAM
#         self.cbam = CBAM3D(dim, ratio=16, kernel_size=7)

#     def build_conv_block(self, dim, norm_layer, use_dropout, use_bias, dilation):
#         conv_block = []
#         p = dilation # Padding = Dilation 以保持尺寸不变
        
#         # First Conv Layer
#         conv_block += [nn.Conv3d(dim, dim, kernel_size=3, padding=p, dilation=dilation, bias=use_bias),
#                        norm_layer(dim),
#                        nn.ReLU(False)]
#         if use_dropout:
#             conv_block += [nn.Dropout(0.5)]

#         # Second Conv Layer
#         conv_block += [nn.Conv3d(dim, dim, kernel_size=3, padding=p, dilation=dilation, bias=use_bias),
#                        norm_layer(dim)]

#         return nn.Sequential(*conv_block)

#     def forward(self, x):
#         residual = self.conv_block(x)
        
#         # [Exp 52]: 在残差路径上应用 CBAM
#         residual = self.cbam(residual)
        
#         out = x + residual
#         return self.relu(out)


# class PixelAwareAttention(nn.Module):
#     """
#     [Exp 30 核心]: 支持 temperature 锐化的 Pixel Attention
#     """
#     def __init__(self, in_channels, is_3d=True, temperature=1.0):
#         super(PixelAwareAttention, self).__init__()
#         Conv = nn.Conv3d if is_3d else nn.Conv2d
#         self.conv1 = Conv(in_channels, 1, kernel_size=1, bias=False)
#         self.sigmoid = nn.Sigmoid()
#         self.temperature = temperature

#     def forward(self, x):
#         attn_map = self.sigmoid(self.conv1(x) * self.temperature)
#         return x * attn_map


# class LocalAwareAttention(nn.Module):
#     """Local Aware Attention (保持不变)"""
#     def __init__(self, F_g, F_l, F_int, is_3d=True):
#         super(LocalAwareAttention, self).__init__()
#         Conv = nn.Conv3d if is_3d else nn.Conv2d
#         Bn = nn.BatchNorm3d if is_3d else nn.BatchNorm2d
        
#         self.W_g = nn.Sequential(
#             Conv(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
#             Bn(F_int)
#         )
#         self.W_x = nn.Sequential(
#             Conv(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
#             Bn(F_int)
#         )
#         self.psi = nn.Sequential(
#             Conv(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
#             Bn(1),
#             nn.Sigmoid()
#         )
#         self.relu = nn.ReLU(inplace=False)

#     def forward(self, g, x):
#         g1 = self.W_g(g)
#         x1 = self.W_x(x)
#         if g1.shape[2:] != x1.shape[2:]:
#             g1 = torch.nn.functional.interpolate(g1, size=x1.shape[2:], mode='trilinear', align_corners=False)
#         psi = self.relu(g1 + x1)
#         psi = self.psi(psi)
#         return x * psi

# class ASPP3D(nn.Module):
#     """
#     [Exp 51 修复版]: 降低膨胀率，防止在小特征图上出现空洞
#     Rates: [1, 2, 3] -> 安全适配 8x8 特征图
#     """
#     def __init__(self, in_channels, out_channels, norm_layer=nn.BatchNorm3d):
#         super(ASPP3D, self).__init__()
        
#         # Branch 1: Dilation 1 (1x1 conv)
#         self.branch1 = nn.Sequential(
#             nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False),
#             norm_layer(out_channels),
#             nn.LeakyReLU(0.2, True)
#         )
        
#         # Branch 2: Dilation 2
#         self.branch2 = nn.Sequential(
#             nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=2, dilation=2, bias=False),
#             norm_layer(out_channels),
#             nn.LeakyReLU(0.2, True)
#         )
        
#         # Branch 3: Dilation 3
#         self.branch3 = nn.Sequential(
#             nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=3, dilation=3, bias=False),
#             norm_layer(out_channels),
#             nn.LeakyReLU(0.2, True)
#         )
        
#         # Branch 5: Global Pooling
#         self.branch5_conv = nn.Sequential(
#             nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False),
#             norm_layer(out_channels),
#             nn.LeakyReLU(0.2, True)
#         )

#         # Fusion
#         self.conv_cat = nn.Sequential(
#             nn.Conv3d(out_channels * 4, out_channels, kernel_size=1, bias=False),
#             norm_layer(out_channels),
#             nn.LeakyReLU(0.2, True)
#         )

#     def forward(self, x):
#         [b, c, d, h, w] = x.size()
        
#         conv1x1 = self.branch1(x)
#         conv3x3_d2 = self.branch2(x)
#         conv3x3_d3 = self.branch3(x)
        
#         # Global Pooling
#         global_feature = F.adaptive_avg_pool3d(x, (1, 1, 1))
#         global_feature = self.branch5_conv(global_feature)
#         global_feature = F.interpolate(global_feature, size=(d, h, w), mode='trilinear', align_corners=True)
        
#         # Concat
#         feature_cat = torch.cat([conv1x1, conv3x3_d2, conv3x3_d3, global_feature], dim=1)
        
#         result = self.conv_cat(feature_cat)
#         return result   

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
#         enc_res_block = ResnetBlock3D(inner_nc, norm_layer, use_dropout, use_bias, dilation=dilation)
#         dec_res_block = ResnetBlock3D(outer_nc, norm_layer, use_dropout, use_bias, dilation=dilation)

#         # Upsample (Upsample + Conv)
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
            
#             # [ASPP Logic]
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

# # ==========================================================
# # Part 2: 遗留兼容组件 (Standard Unet) - [必须保留以防报错]
# # ==========================================================

# class UnetGenerator(nn.Module):
#     """原始 Standard U-Net Generator (用于兼容旧代码引用)"""
#     def __init__(self, input_nc, output_nc, num_downs, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, is_3d=False, use_attention=False):
#         super(UnetGenerator, self).__init__()
#         # 构造 unet 结构
#         unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True, is_3d=is_3d, use_attention=False)
#         for i in range(num_downs - 5): 
#             unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout, is_3d=is_3d, use_attention=use_attention)
#         unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, is_3d=is_3d, use_attention=use_attention)
#         unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer, is_3d=is_3d, use_attention=use_attention)
#         unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer, is_3d=is_3d, use_attention=use_attention)
#         self.model = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer, is_3d=is_3d, use_attention=use_attention)

#     def forward(self, input):
#         return self.model(input)

# class UnetSkipConnectionBlock(nn.Module):
#     """原始 Unet Block (用于兼容 UnetGenerator)"""
#     def __init__(self, outer_nc, inner_nc, input_nc=None,
#                  submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False, is_3d=False, use_attention=False):
#         super(UnetSkipConnectionBlock, self).__init__()
#         self.outermost = outermost
#         self.innermost = innermost
#         self.use_attention = use_attention
#         self.is_3d = is_3d
#         if input_nc is None: input_nc = outer_nc
#         Conv = nn.Conv3d if is_3d else nn.Conv2d
#         ConvTrans = nn.ConvTranspose3d if is_3d else nn.ConvTranspose2d
        
#         downconv = Conv(input_nc, inner_nc, kernel_size=4, stride=2, padding=1, bias=False)
#         downrelu = nn.LeakyReLU(0.2, False)
#         downnorm = norm_layer(inner_nc)
#         uprelu = nn.ReLU(False)
#         upnorm = norm_layer(outer_nc)

#         if outermost:
#             upconv = ConvTrans(inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1)
#             down = [downconv]
#             up = [uprelu, upconv, nn.Tanh()]
#             model = down + [submodule] + up
#         elif innermost:
#             upconv = ConvTrans(inner_nc, outer_nc, kernel_size=4, stride=2, padding=1, bias=False)
#             down = [downrelu, downconv]
#             up = [uprelu, upconv, upnorm]
#             model = down + up
#         else:
#             upconv = ConvTrans(inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1, bias=False)
#             down = [downrelu, downconv, downnorm]
#             up = [uprelu, upconv, upnorm]
#             if use_dropout: model = down + [submodule] + up + [nn.Dropout(0.5)]
#             else: model = down + [submodule] + up
#         self.model = nn.Sequential(*model)
        
#         if self.use_attention and not self.innermost:
#             self.pa = PixelAwareAttention(input_nc, is_3d=is_3d)
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

# # 兼容性 Wrapper (用于 netG=='unet_3d' 时)
# class UnetGenerator3D(UnetGenerator):
#     def __init__(self, input_nc, output_nc, num_downs, ngf=64, norm_layer=nn.BatchNorm3d, use_dropout=False, use_attention=False):
#         super(UnetGenerator3D, self).__init__(input_nc, output_nc, num_downs, ngf, norm_layer, use_dropout, is_3d=True, use_attention=use_attention)

