"""
数据加载模块的初始化文件。
包含 create_dataset 工厂函数，对外提供统一的数据接口。
"""
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from data.ultrasound_dataset import UltrasoundDataset

def create_dataset(opt):
    """
    创建并返回一个 PyTorch DataLoader。
    
    参数:
        opt: 配置参数对象
    """
    # 1. 实例化数据集
    dataset = UltrasoundDataset(opt)
    
    # 2. 初始化采样相关变量
    shuffle = True if opt.isTrain else False
    sampler = None

    # 3. [新增逻辑] 权重采样检测
    # 只有在训练模式且数据集支持权重计算时才启用
    if opt.isTrain and hasattr(dataset, 'get_sample_weights'):
        print("⚖️  Detecting Class Imbalance... Calculating weights.")
        try:
            # 获取权重 (调用我们在 UltrasoundDataset 里新写的方法)
            sample_weights = dataset.get_sample_weights()
            
            # 创建加权采样器
            # replacement=True 允许重复抽样，这对平衡小样本至关重要
            sampler = WeightedRandomSampler(
                weights=sample_weights,
                num_samples=len(dataset),
                replacement=True
            )
            
            # 【关键规则】如果使用了 sampler，shuffle 必须为 False
            shuffle = False
            print("✅  WeightedRandomSampler activated. (Shuffle set to False automatically)")
            
        except Exception as e:
            print(f"⚠️  Failed to calculate weights: {e}. Fallback to standard shuffling.")
            sampler = None
            shuffle = True

    # 4. 创建 DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=opt.batch_size,
        shuffle=shuffle,        # 这里由上面的逻辑决定 (如果用了 sampler 就是 False)
        sampler=sampler,        # 注入采样器 (如果没启用就是 None)
        num_workers=opt.num_workers,
        pin_memory=True,
        drop_last=True if opt.isTrain else False
    )
    
    print(f"Dataset [{type(dataset).__name__}] created. "
          f"Size: {len(dataset)}, Batch Size: {opt.batch_size}, "
          f"Shuffle: {shuffle}, Sampler: {'Active' if sampler else 'None'}, "
          f"Workers: {opt.num_workers}")
          
    return dataloader