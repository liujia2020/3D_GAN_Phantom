import torch
import torch.nn as nn
import functools

class NLayerDiscriminator3D(nn.Module):
    """
    3D PatchGAN 判别器。
    
    设计理念：
    - 不输出单一的 True/False，而是输出一个 3D 矩阵 (Patch Map)。
    - 矩阵中的每个点代表原图中一小块 3D 区域的真假。
    - 这种结构对高频纹理（如超声散斑）特别敏感。
    """

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.InstanceNorm3d):
        """
        参数:
            input_nc (int):  输入通道数 (通常是 2 = 1个LQ + 1个HQ，或者是生成图)
            ndf (int):       第一层卷积的滤波器数量
            n_layers (int):  判别器的深度 (默认 3 层)
            norm_layer:      归一化层类型
        """
        super(NLayerDiscriminator3D, self).__init__()
        
        # 确定是否需要 Bias
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm3d
        else:
            use_bias = norm_layer == nn.InstanceNorm3d

        kw = 4 # Kernel Size
        padw = 1 # Padding
        
        sequence = []
        
        # --- 第 0 层 (输入层) ---
        # 这一层通常不加 Norm，直接 LeakyReLU
        sequence += [nn.Conv3d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), 
                     nn.LeakyReLU(0.2, True)]
        
        # --- 中间层 (Hidden Layers) ---
        # 循环构建 n_layers 层，每层通道数翻倍 (ndf -> ndf*2 -> ndf*4 ...)
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8) # 限制最大通道倍数为 8
            sequence += [
                nn.Conv3d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        # --- 倒数第二层 ---
        # 这一层 Stride=1，不改变尺寸，只改变通道特征
        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv3d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        # --- 输出层 ---
        # 输出单通道 (1 channel) 的预测图
        sequence += [nn.Conv3d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]
        
        # 注意：这里最后不加 Sigmoid，因为我们使用 BCEWithLogitsLoss (数值更稳定)
        
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        return self.model(input)