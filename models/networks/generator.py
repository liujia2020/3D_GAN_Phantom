import torch
import torch.nn as nn
import torch.nn.functional as F

class UnetGenerator3D(nn.Module):
    """
    [极简版 U-Net 3D]
    去除所有 Attention 模块。
    回归最经典的 Skip Connection (Concat)。
    这是训练稀疏点靶的最稳健架构。
    """
    def __init__(self, input_nc, output_nc, num_downs, ngf=64, norm_layer=nn.BatchNorm3d, use_dropout=False):
        super(UnetGenerator3D, self).__init__()
        
        # 构造 Encoder (下采样)
        # Layer 1: Input -> 64
        self.enc1 = nn.Conv3d(input_nc, ngf, 4, 2, 1) 
        
        # Layer 2: 64 -> 128
        self.enc2_block = self._block(ngf, ngf*2, norm_layer)
        
        # Layer 3: 128 -> 256
        self.enc3_block = self._block(ngf*2, ngf*4, norm_layer)
        
        # Layer 4: 256 -> 512
        self.enc4_block = self._block(ngf*4, ngf*8, norm_layer)
        
        # Layer 5 (Bottleneck): 512 -> 512 
        self.enc5_block = self._block(ngf*8, ngf*8, norm_layer, innermost=True)

        # 构造 Decoder (上采样)
        # Dec 5: 512 -> 512
        self.dec5 = self._up_block(ngf*8, ngf*8, norm_layer)
        
        # Dec 4: 1024 (512+512) -> 256
        self.dec4 = self._up_block(ngf*8 * 2, ngf*4, norm_layer)
        
        # Dec 3: 512 (256+256) -> 128
        self.dec3 = self._up_block(ngf*4 * 2, ngf*2, norm_layer)
        
        # Dec 2: 256 (128+128) -> 64
        self.dec2 = self._up_block(ngf*2 * 2, ngf, norm_layer)
        
        # Dec 1: 128 (64+64) -> Output
        self.dec1 = nn.Sequential(
            nn.Conv3d(ngf * 2, ngf, 3, 1, 1),
            nn.ReLU(True),
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False),
            # 最后一层映射回 Output Channel
            nn.Conv3d(ngf, output_nc, 3, 1, 1),
            nn.Tanh() # 输出范围 [-1, 1]
        )

    def _block(self, in_c, out_c, norm_layer, innermost=False):
        layers = [nn.LeakyReLU(0.2, True), nn.Conv3d(in_c, out_c, 4, 2, 1, bias=False)]
        if not innermost:
            layers.append(norm_layer(out_c))
        return nn.Sequential(*layers)

    def _up_block(self, in_c, out_c, norm_layer):
        return nn.Sequential(
            nn.Conv3d(in_c, out_c, 3, 1, 1, bias=False),
            norm_layer(out_c),
            nn.ReLU(True),
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        )

    def forward(self, x):
        # --- Encoder ---
        e1 = self.enc1(x)           # 64
        e2 = self.enc2_block(e1)    # 128
        e3 = self.enc3_block(e2)    # 256
        e4 = self.enc4_block(e3)    # 512
        e5 = self.enc5_block(e4)    # 512 (Bottleneck)
        
        # --- Decoder (Direct Skip Connection) ---
        d5 = self.dec5(e5)                          # 512
        d4 = self.dec4(torch.cat([d5, e4], 1))      # 512+512 -> 256
        d3 = self.dec3(torch.cat([d4, e3], 1))      # 256+256 -> 128
        d2 = self.dec2(torch.cat([d3, e2], 1))      # 128+128 -> 64
        d1 = self.dec1(torch.cat([d2, e1], 1))      # 64+64   -> Output
        
        return d1