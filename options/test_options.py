from .base_options import BaseOptions

class TestOptions(BaseOptions):
    """
    此类包含仅在测试期间使用的参数。
    它同样继承自 BaseOptions。
    """

    def initialize(self, parser):
        # 1. 加载通用参数
        parser = BaseOptions.initialize(self, parser)
        
        # --- 测试特有参数 ---
        parser.add_argument('--results_dir', type=str, default='./results/', help='保存测试结果的目录')
        parser.add_argument('--phase', type=str, default='test', help='当前阶段 [train, val, test]')
        
        # 评估模式
        parser.add_argument('--eval', action='store_true', help='如果指定，在测试期间将网络设置为 eval 模式 (这会影响 Dropout 和 BatchNorm)')
        parser.add_argument('--num_test', type=int, default=float('inf'), help='测试多少张图片？默认是全部')
        
        # [物理相关]
        # 有时候我们只想生成 NIfTI 体数据，不想保存切片图（节省时间）
        parser.add_argument('--save_vol_only', action='store_true', help='如果指定，仅保存 .nii.gz 体数据，不保存 2D 切片图')

        # rewrite devalue values
        parser.set_defaults(model='test')
        # 测试时不需要加载优化器参数，也不需要计算梯度，所以这里我们通过 phase 区分
        self.isTrain = False
        return parser