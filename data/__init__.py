"""
data/__init__.py
"""
import importlib
import torch.utils.data
from data.base_dataset import BaseDataset
from data.ultrasound_dataset import UltrasoundDataset

def create_dataset(opt):
    """Create a dataset given the option."""
    dataset = UltrasoundDataset(opt)
    print("dataset [UltrasoundDataset] was created")
    
    # 依据参数决定是否打乱
    shuffle = True if opt.isTrain and not opt.serial_batches else False
    
    # [关键修复]: 在训练阶段丢弃最后一个不完整的 batch，防止 BatchNorm 在最深层 1x1x1 时崩溃
    drop_last = True if opt.isTrain else False
    
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=opt.batch_size,
        shuffle=shuffle,
        num_workers=int(opt.num_threads),
        drop_last=drop_last  # [新增参数]
    )
    return data_loader

def get_option_setter(dataset_name):
    """Return the static method <modify_commandline_options> of the dataset class."""
    # 直接返回 UltrasoundDataset 的方法，因为现在 BaseDataset 里已经有了，不会报错
    return UltrasoundDataset.modify_commandline_options