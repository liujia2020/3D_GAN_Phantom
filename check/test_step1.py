import torch
import torch.nn as nn
from models.losses import TVLoss, FrequencyLoss, VGG19FeatureExtractor
from models.networks.blocks import ResnetBlock3D, ASPP3D, CBAM3D, PixelAwareAttention

def test_losses():
    print("Testing Losses...")
    # 模拟数据 [Batch, Channel, Depth, Height, Width]
    pred = torch.randn(2, 1, 16, 64, 64)
    target = torch.randn(2, 1, 16, 64, 64)
    
    # 1. TV Loss
    tv_loss = TVLoss()
    loss = tv_loss(pred)
    print(f"  TV Loss check: {loss.item():.4f} (Success)")
    
    # 2. FFL Loss
    ffl_loss = FrequencyLoss()
    loss = ffl_loss(pred, target)
    print(f"  FFL Loss check: {loss.item():.4f} (Success)")
    
    # 3. VGG (需要 RGB 3通道)
    vgg = VGG19FeatureExtractor()
    rgb_img = torch.randn(4, 3, 64, 64) # VGG处理的是2D切片
    feat = vgg(rgb_img)
    print(f"  VGG Feature shape: {feat.shape} (Success)")

def test_blocks():
    print("\nTesting Blocks...")
    x = torch.randn(2, 64, 16, 32, 32) # [B, C, D, H, W]
    norm = nn.BatchNorm3d
    
    # 1. ResnetBlock3D (With CBAM)
    block = ResnetBlock3D(dim=64, norm_layer=norm, use_dropout=False, use_bias=False)
    out = block(x)
    print(f"  ResnetBlock3D output: {out.shape} (Success)")
    
    # 2. ASPP3D
    aspp = ASPP3D(in_channels=64, out_channels=64, norm_layer=norm)
    out = aspp(x)
    print(f"  ASPP3D output: {out.shape} (Success)")
    
    # 3. PixelAwareAttention
    pa = PixelAwareAttention(in_channels=64, is_3d=True)
    out = pa(x)
    print(f"  PixelAwareAttention output: {out.shape} (Success)")

if __name__ == '__main__':
    try:
        test_losses()
        test_blocks()
        print("\n>>> Step 1 测试通过！模块拆分成功。")
    except Exception as e:
        print(f"\n>>> Step 1 测试失败！错误信息:\n{e}")
        exit(1)