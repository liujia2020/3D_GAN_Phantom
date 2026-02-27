import sys
import torch
from options.train_options import TrainOptions
from models import create_model

# 1. 模拟命令行参数，防止报错
# 给定一个假路径和假名称，确保 TrainOptions 能够解析通过
sys.argv = [
    'test_structure.py', 
    '--dataroot', './mock_data', 
    '--name', 'test_reconstruct',
    '--gpu_ids', '-1'  # 强制使用 CPU 进行结构测试
]

# 2. 解析参数并初始化模型
try:
    opt = TrainOptions().parse()
    model = create_model(opt)
    print("✅ 模型实例化成功")

    # 3. 模拟输入数据 (Batch_size, Channel, Depth, Height, Width)
    # 使用你提到的当前尺寸 128*64*64
    mock_data = {
        'lq': torch.randn(1, 1, 128, 64, 64),
        'sq': torch.randn(1, 1, 128, 64, 64),
        'lq_path': ['mock_path/test.nii']
    }

    # 4. 运行前向链路验证命名
    model.set_input(mock_data)
    model.forward()
    model.compute_visuals()

    # 5. 严格检查命名是否符合我们的约定
    assert hasattr(model, 'input_lq'), "❌ 命名错误: input_lq 缺失"
    assert hasattr(model, 'output_gen'), "❌ 命名错误: output_gen 缺失"
    assert hasattr(model, 'target_hq'), "❌ 命名错误: target_hq 缺失"
    
    # 验证可视化切片是否存在
    visuals = model.get_current_visuals()
    if 'input_lq_slice' in visuals:
        print(f"✅ 命名统一测试通过！切片尺寸: {visuals['input_lq_slice'].shape}")
    else:
        print("❌ 错误: visual_names 映射不正确")

except Exception as e:
    print(f"❌ 测试中途崩溃: {str(e)}")
    import traceback
    traceback.print_exc()