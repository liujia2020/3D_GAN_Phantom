import torch
import torch.nn as nn
import functools

# ==========================================================
# 1. 注意力模块定义 (Pixel Attention & Local Aware Attention)
# ==========================================================

class PixelAwareAttention(nn.Module):
    """
    Pixel Attention (PA): 
    关注 'Where'。通过空间注意力机制压制背景噪声（灰纱），保留前景（点靶）。
    """
    def __init__(self, in_channels, is_3d=True):
        super(PixelAwareAttention, self).__init__()
        Conv = nn.Conv3d if is_3d else nn.Conv2d
        
        # 1x1 卷积压缩信息，生成空间掩码
        self.conv1 = Conv(in_channels, 1, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: [B, C, D, H, W]
        attn_map = self.sigmoid(self.conv1(x))
        return x * attn_map

class LocalAwareAttention(nn.Module):
    """
    Local Aware Attention (LAA) / Attention Gate:
    关注 'What'。利用 Decoder 的深层特征（g）作为门控，
    去筛选 Encoder 的浅层特征（x）中真正重要的细节。
    这能有效防止微小点靶被'吞噬'。
    """
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
        
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        # g: Decoder feature (Gating signal)
        # x: Encoder feature (Skip connection)
        
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        
        # 尺寸对齐 (以防万一有微小差异，虽然 UNet 通常是对齐的)
        if g1.shape[2:] != x1.shape[2:]:
            g1 = torch.nn.functional.interpolate(g1, size=x1.shape[2:], mode='trilinear', align_corners=False)
        
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        
        return x * psi

# ==========================================================
# 2. 生成器主类
# ==========================================================

class UnetGenerator(nn.Module):
    """Create a Unet-based generator"""

    def __init__(self, input_nc, output_nc, num_downs, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, is_3d=False, use_attention=False):
        
        """Construct a Unet generator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            output_nc (int) -- the number of channels in output images
            num_downs (int) -- the number of downsamplings in UNet. For example, # if |num_downs| == 7,
                                image of size 128x128 will become of size 1x1 # at the bottleneck
            ngf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer
            is_3d (bool)    -- if use 3d generator
            use_attention (bool) -- 是否开启注意力机制 (PA + LAA)
        """
        super(UnetGenerator, self).__init__()
        # construct unet structure
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True, is_3d=is_3d, use_attention=False)  # 最内层通常不加 Attention
        
        for i in range(num_downs - 5): # add intermediate layers with ngf * 8 filters
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout, is_3d=is_3d, use_attention=use_attention)
        
        # gradually reduce the number of filters from ngf * 8 to ngf
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, is_3d=is_3d, use_attention=use_attention)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer, is_3d=is_3d, use_attention=use_attention)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer, is_3d=is_3d, use_attention=use_attention)
        
        self.model = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer, is_3d=is_3d, use_attention=use_attention)  # outermost layer

    def forward(self, input):
        """Standard forward"""
        return self.model(input)


class UnetSkipConnectionBlock(nn.Module):
    """Defines the Unet submodule with skip connection.
    X -------------------identity----------------------
    |-- downsampling -- |submodule| -- upsampling --|
    """

    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False, is_3d=False, use_attention=False):
        """Construct a Unet submodule with skip connections.
        Parameters:
            outer_nc (int) -- the number of filters in the outer conv layer
            inner_nc (int) -- the number of filters in the inner conv layer
            input_nc (int) -- the number of channels in input images/features
            submodule (UnetSkipConnectionBlock) -- previously defined submodule
            outermost (bool)    -- if this module is the outermost module
            innermost (bool)    -- if this module is the innermost module
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
            is_3d (bool)        -- if use 3d conv
            use_attention (bool)-- [New] switch for Attention Mechanism
        """
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        self.innermost = innermost
        self.use_attention = use_attention
        self.is_3d = is_3d

        if input_nc is None:
            input_nc = outer_nc
        
        Conv = nn.Conv3d if is_3d else nn.Conv2d
        ConvTrans = nn.ConvTranspose3d if is_3d else nn.ConvTranspose2d
        
        downconv = Conv(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=False)
        
        # 定义下采样部分
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = ConvTrans(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = ConvTrans(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=False)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = ConvTrans(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=False)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

        # ====================================================
        # [核心修改] 初始化注意力模块 (PA + LAA)
        # ====================================================
        if self.use_attention and not self.innermost:
            # 1. Pixel Attention (PA): 作用于 Encoder 的 Skip Connection (x)
            self.pa = PixelAwareAttention(input_nc, is_3d=is_3d)
            
            # 2. Local Aware Attention (LAA): 作用于融合阶段
            # Gating Signal (g) 来自 Decoder (self.model(x))，通道数是 outer_nc
            # Skip Signal (x) 来自 Encoder，通道数是 input_nc
            self.laa = LocalAwareAttention(F_g=outer_nc, F_l=input_nc, F_int=max(input_nc//2, 1), is_3d=is_3d)


    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:   # add skip connections
            
            # 1. 正常的 U-Net 路径
            decoder_feature = self.model(x) # 这是经过下采样+子模块+上采样的结果 (Decoder Feature)
            skip_feature = x                # 这是当前的输入 (Encoder Feature / Skip Connection)

            # 2. 注意力机制处理 (如果开启)
            if self.use_attention and not self.innermost:
                # Step A: 像素注意力 (Pixel Attention) -> 去灰纱
                # 对 Skip Connection 进行空间加权，压制背景
                skip_feature = self.pa(skip_feature)
                
                # Step B: 局部感知注意力 (Local Aware Attention) -> 防吞噬
                # 利用 Decoder 的信息来指导 Skip Connection，只保留重要的细节
                skip_feature = self.laa(g=decoder_feature, x=skip_feature)
            
            # 3. 拼接
            return torch.cat([skip_feature, decoder_feature], 1)
# ==========================================================
# 3. 兼容性 Alias (修复 ImportError)
# ==========================================================
class UnetGenerator3D(UnetGenerator):
    """
    Wrapper for 3D Unet Generator to be compatible with existing code imports.
    Forces is_3d=True.
    """
    def __init__(self, input_nc, output_nc, num_downs, ngf=64, norm_layer=nn.BatchNorm3d, use_dropout=False, use_attention=False):
        # 强制设置 is_3d=True
        super(UnetGenerator3D, self).__init__(input_nc, output_nc, num_downs, ngf, norm_layer, use_dropout, is_3d=True, use_attention=use_attention)

# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# class UnetGenerator3D(nn.Module):
#     """
#     [极简版 U-Net 3D]
#     去除所有 Attention 模块。
#     回归最经典的 Skip Connection (Concat)。
#     这是训练稀疏点靶的最稳健架构。
#     """
#     def __init__(self, input_nc, output_nc, num_downs, ngf=64, norm_layer=nn.BatchNorm3d, use_dropout=False):
#         super(UnetGenerator3D, self).__init__()
        
#         # 构造 Encoder (下采样)
#         # Layer 1: Input -> 64
#         self.enc1 = nn.Conv3d(input_nc, ngf, 4, 2, 1) 
        
#         # Layer 2: 64 -> 128
#         self.enc2_block = self._block(ngf, ngf*2, norm_layer)
        
#         # Layer 3: 128 -> 256
#         self.enc3_block = self._block(ngf*2, ngf*4, norm_layer)
        
#         # Layer 4: 256 -> 512
#         self.enc4_block = self._block(ngf*4, ngf*8, norm_layer)
        
#         # Layer 5 (Bottleneck): 512 -> 512 
#         self.enc5_block = self._block(ngf*8, ngf*8, norm_layer, innermost=True)

#         # 构造 Decoder (上采样)
#         # Dec 5: 512 -> 512
#         self.dec5 = self._up_block(ngf*8, ngf*8, norm_layer)
        
#         # Dec 4: 1024 (512+512) -> 256
#         self.dec4 = self._up_block(ngf*8 * 2, ngf*4, norm_layer)
        
#         # Dec 3: 512 (256+256) -> 128
#         self.dec3 = self._up_block(ngf*4 * 2, ngf*2, norm_layer)
        
#         # Dec 2: 256 (128+128) -> 64
#         self.dec2 = self._up_block(ngf*2 * 2, ngf, norm_layer)
        
#         # Dec 1: 128 (64+64) -> Output
#         self.dec1 = nn.Sequential(
#             nn.Conv3d(ngf * 2, ngf, 3, 1, 1),
#             nn.ReLU(True),
#             nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False),
#             # 最后一层映射回 Output Channel
#             nn.Conv3d(ngf, output_nc, 3, 1, 1),
#             nn.Tanh() # 输出范围 [-1, 1]
#         )

#     def _block(self, in_c, out_c, norm_layer, innermost=False):
#         layers = [nn.LeakyReLU(0.2, True), nn.Conv3d(in_c, out_c, 4, 2, 1, bias=False)]
#         if not innermost:
#             layers.append(norm_layer(out_c))
#         return nn.Sequential(*layers)

#     def _up_block(self, in_c, out_c, norm_layer):
#         return nn.Sequential(
#             nn.Conv3d(in_c, out_c, 3, 1, 1, bias=False),
#             norm_layer(out_c),
#             nn.ReLU(True),
#             nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
#         )

#     def forward(self, x):
#         # --- Encoder ---
#         e1 = self.enc1(x)           # 64
#         e2 = self.enc2_block(e1)    # 128
#         e3 = self.enc3_block(e2)    # 256
#         e4 = self.enc4_block(e3)    # 512
#         e5 = self.enc5_block(e4)    # 512 (Bottleneck)
        
#         # --- Decoder (Direct Skip Connection) ---
#         d5 = self.dec5(e5)                          # 512
#         d4 = self.dec4(torch.cat([d5, e4], 1))      # 512+512 -> 256
#         d3 = self.dec3(torch.cat([d4, e3], 1))      # 256+256 -> 128
#         d2 = self.dec2(torch.cat([d3, e2], 1))      # 128+128 -> 64
#         d1 = self.dec1(torch.cat([d2, e1], 1))      # 64+64   -> Output
        
#         return d1