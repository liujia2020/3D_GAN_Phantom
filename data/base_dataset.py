"""
data/base_dataset.py
"""
import torch.utils.data as data
from abc import ABC, abstractmethod
import torch

class BaseDataset(data.Dataset, ABC):
    """This class is an abstract base class (ABC) for datasets."""

    def __init__(self, opt):
        """Initialize the class; save the options in the class"""
        self.opt = opt
        self.root = opt.dataroot

    @staticmethod
    def modify_commandline_options(parser, is_train):
        """Add new dataset-specific options, and rewrite default values for existing options.
        这是导致 AttributeError 的关键缺失方法。
        """
        return parser

    @abstractmethod
    def __len__(self):
        """Return the total number of images in the dataset."""
        return 0

    @abstractmethod
    def __getitem__(self, index):
        """Return a data point and its metadata information."""
        pass