from .base_options import BaseOptions

class TrainOptions(BaseOptions):
    """
    此类仅包含训练阶段特有的参数。
    它继承自 BaseOptions，所以你在 base 里定义的参数（如 patch_size），这里都能用。
    """
    
    def initialize(self, parser):
        # 1. 先加载父类 (BaseOptions) 的通用参数
        parser = BaseOptions.initialize(self, parser)
        
        # --- 训练可视化与日志 (Visualization & Logging) ---
        parser.add_argument('--display_freq', type=int, default=100, help='每隔多少次迭代(iteration)在屏幕/日志中显示一次结果')
        parser.add_argument('--print_freq', type=int, default=20, help='每隔多少次迭代在终端打印一次 Loss')
        parser.add_argument('--save_latest_freq', type=int, default=1000, help='每隔多少次迭代保存一次 "latest" 模型 (防崩溃)')
        parser.add_argument('--save_epoch_freq', type=int, default=5, help='每隔多少个 epoch 保存一次模型检查点')
        parser.add_argument('--save_by_iter', action='store_true', help='如果开启，模型将按迭代次数保存，而不是按 epoch 保存')
        
        # --- 训练周期与策略 (Training Schedule) ---
        parser.add_argument('--continue_train', action='store_true', help='如果指定，程序会尝试加载最新的 checkpoints 继续训练')
        parser.add_argument('--epoch_count', type=int, default=1, help='起始 epoch 计数 (主要用于断点续训时调整显示)')
        parser.add_argument('--phase', type=str, default='train', help='当前阶段 [train, val, test] (数据加载器会用到)')
        
        # 核心：epoch 策略
        parser.add_argument('--n_epochs', type=int, default=100, help='以固定初始学习率训练的 epoch 数')
        parser.add_argument('--n_epochs_decay', type=int, default=100, help='学习率线性衰减到 0 的 epoch 数')
        
        # --- 优化器参数 (Optimizer) ---
        parser.add_argument('--beta1', type=float, default=0.5, help='Adam 优化器的 momentum term beta1')
        parser.add_argument('--lr', type=float, default=0.0002, help='生成器 (G) 的初始学习率')
        
        parser.add_argument('--lr_d_ratio', type=float, default=1.0, help='D 的学习率相对于 G 的比率 (例如 0.5 代表 D_lr = 0.5 * G_lr)')
        
        parser.add_argument('--lr_policy', type=str, default='linear', help='学习率衰减策略 [linear | step | plateau | cosine]')
        parser.add_argument('--lr_decay_iters', type=int, default=50, help='每多少步衰减一次学习率 (仅针对 step 策略)')
        
        # --- GAN 损失函数选择 ---
        parser.add_argument('--gan_mode', type=str, default='vanilla', help='GAN 损失类型 [vanilla | lsgan | wgangp]')
        parser.add_argument('--pixel_loss', type=str, default='L1', choices=['L1', 'L2'], help='像素级损失函数的类型: L1 (MAE) 或 L2 (MSE)')
        parser.add_argument('--lambda_pixel', type=float, default=10.0, help='weight for pixel loss')
        # [新增] 边缘损失权重
        parser.add_argument('--lambda_edge', type=float, default=10.0, help='weight for edge loss')
        # [新增] 感知损失权重
        parser.add_argument('--lambda_perceptual', type=float, default=0.2, help='weight for perceptual loss')
        # ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓
        # [关键修复] 补上 pool_size (图像缓冲池大小，默认为50)
        parser.add_argument('--pool_size', type=int, default=50, help='the size of image buffer that stores previously generated images')
        # ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑

        self.isTrain = True
        return parser