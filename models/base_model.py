import os
import torch
from collections import OrderedDict
from abc import ABC, abstractmethod
from torch.optim import lr_scheduler

class BaseModel(ABC):
    """所有模型的抽象基类"""

    def __init__(self, opt):
        """
        初始化 BaseModel。
        参数:
            opt: 命令行参数对象
        """
        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.isTrain = opt.isTrain
        
        # 自动获取设备
        self.device = torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')
        
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)  # 模型保存路径
        
        # 下面这些列表需要在子类 (AuganModel) 中填充
        self.loss_names = []      # e.g., ['G_GAN', 'D_Real']
        self.model_names = []     # e.g., ['G', 'D']
        self.visual_names = []    # e.g., ['real_lq', 'fake_hq']
        self.optimizers = []      # 存放优化器
        self.image_paths = []     # 存放当前处理的图片路径
        self.metric = 0           #用于保存最好的模型

    @staticmethod
    def modify_commandline_options(parser, is_train):
        """添加特定于模型的选项 (默认不做任何事)"""
        return parser

    @abstractmethod
    def set_input(self, input):
        """从数据加载器解包输入数据"""
        pass

    @abstractmethod
    def forward(self):
        """运行前向传播"""
        pass

    @abstractmethod
    def optimize_parameters(self):
        """计算损失、梯度并更新权重"""
        pass

    def setup(self, opt):
        """加载与打印网络: 
        1. 加载权重 (如果是继续训练或测试)
        2. 初始化学习率调度器 (schedulers)
        3. 打印网络架构信息
        """
        if self.isTrain:
            self.schedulers = [get_scheduler(optimizer, opt) for optimizer in self.optimizers]
            
        if not self.isTrain or opt.continue_train:
            load_suffix = 'iter_%d' % opt.load_iter if opt.load_iter > 0 else opt.epoch
            self.load_networks(load_suffix)
            
        self.print_networks(opt.verbose)

    def eval(self):
        """将模型设置为测试模式 (影响 Dropout 和 BatchNorm)"""
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                net.eval()

    def train(self):
        """将模型设置为训练模式"""
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                net.train()

    def test(self):
        """测试阶段的前向传播 (关闭梯度计算)"""
        with torch.no_grad():
            self.forward()

    def update_learning_rate(self):
        """更新所有优化器的学习率 (在每个 epoch 结束时调用)"""
        old_lr = self.optimizers[0].param_groups[0]['lr']
        for scheduler in self.schedulers:
            if self.opt.lr_policy == 'plateau':
                scheduler.step(self.metric)
            else:
                scheduler.step()
        
        lr = self.optimizers[0].param_groups[0]['lr']
        # print('learning rate %.7f -> %.7f' % (old_lr, lr))

    def get_current_losses(self):
        """返回当前的损失值字典"""
        errors_ret = OrderedDict()
        for name in self.loss_names:
            if isinstance(name, str):
                # errors_ret[name] = float(getattr(self, 'loss_' + name))
                errors_ret[name] = getattr(self, 'loss_' + name).item()
        return errors_ret

    def save_networks(self, epoch):
        """保存模型权重到硬盘"""
        for name in self.model_names:
            if isinstance(name, str):
                save_filename = '%s_net_%s.pth' % (epoch, name)
                save_path = os.path.join(self.save_dir, save_filename)
                net = getattr(self, 'net' + name)

                if len(self.gpu_ids) > 0 and torch.cuda.is_available():
                    # 处理 DataParallel 包装的情况
                    torch.save(net.cpu().state_dict(), save_path)
                    net.cuda(self.gpu_ids[0])
                else:
                    torch.save(net.cpu().state_dict(), save_path)

    def load_networks(self, epoch):
        """从硬盘加载模型权重"""
        for name in self.model_names:
            if isinstance(name, str):
                load_filename = '%s_net_%s.pth' % (epoch, name)
                load_path = os.path.join(self.save_dir, load_filename)
                net = getattr(self, 'net' + name)
                if isinstance(net, torch.nn.DataParallel):
                    net = net.module
                print('loading the model from %s' % load_path)
                state_dict = torch.load(load_path, map_location=self.device)
                if hasattr(state_dict, '_metadata'):
                    del state_dict._metadata
                net.load_state_dict(state_dict)

    def print_networks(self, verbose):
        """打印网络参数量"""
        print('---------- Networks initialized -------------')
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                num_params = 0
                for param in net.parameters():
                    num_params += param.numel()
                if verbose:
                    print(net)
                print('[Network %s] Total number of parameters : %.3f M' % (name, num_params / 1e6))
        print('-----------------------------------------------')

    def set_requires_grad(self, nets, requires_grad=False):
        """设置网络的 requires_grad 标志 (冻结/解冻权重)"""
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

# --- 辅助函数: 学习率调度器 ---

def get_scheduler(optimizer, opt):
    """根据 options 返回学习率调度器"""
    if opt.lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.n_epochs) / float(opt.n_epochs_decay + 1)
            return max(0.0, lr_l)
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.n_epochs, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler