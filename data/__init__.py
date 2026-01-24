"""
data/__init__.py
"""
import importlib
import torch.utils.data
from data.base_dataset import BaseDataset
from data.ultrasound_dataset import UltrasoundDataset

def create_dataset(opt):
    """创建并返回 DataLoader"""
    dataset = UltrasoundDataset(opt)
    print("dataset [UltrasoundDataset] was created")
    
    # 依据参数决定是否打乱
    shuffle = True if opt.isTrain and not opt.serial_batches else False
    
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=opt.batch_size,
        shuffle=shuffle,
        num_workers=int(opt.num_threads))
    return data_loader

def get_option_setter(dataset_name):
    """获取数据集特定的参数设置"""
    # 不管传入什么 dataset_name，强制返回 UltrasoundDataset 的参数设置
    return UltrasoundDataset.modify_commandline_options