import argparse
import os
import util  # 确保是 import util
import torch
import models
import data

class BaseOptions():
    """This class defines options used during both training and test time."""

    def __init__(self):
        """Reset the class; indicates the class hasn't been initailized"""
        self.initialized = False

    def initialize(self, parser):
        parser.add_argument('--dataroot', required=True, help='path to images')
        parser.add_argument('--batch_size', type=int, default=1, help='input batch size')
        parser.add_argument('--loadSize', type=int, default=286, help='scale images to this size')
        parser.add_argument('--cropSize', type=int, default=256, help='then crop to this size')
        parser.add_argument('--input_nc', type=int, default=1, help='# of input image channels')
        parser.add_argument('--output_nc', type=int, default=1, help='# of output image channels')
        
        # === 数据文件夹名称 ===
        parser.add_argument('--dir_lq', type=str, default='Recon_LQ_03', help='folder name for input data (LQ)')
        parser.add_argument('--dir_sq', type=str, default='Recon_SQ_75', help='folder name for target data (SQ)')
        # ==============================
        parser.add_argument('--use_attention', action='store_true', help='if specified, use Attention Mechanism (PA + LAA) in Generator')
        parser.add_argument('--ngf', type=int, default=64, help='# of gen filters in the last conv layer')
        parser.add_argument('--ndf', type=int, default=64, help='# of discrim filters in the first conv layer')
        parser.add_argument('--netD', type=str, default='basic', help='specify discriminator architecture')
        parser.add_argument('--netG', type=str, default='resnet_9blocks', help='specify generator architecture')
        parser.add_argument('--n_layers_D', type=int, default=3, help='only used if netD==n_layers')
        parser.add_argument('--norm', type=str, default='instance', help='instance normalization or batch normalization')
        parser.add_argument('--init_type', type=str, default='normal', help='network initialization')
        parser.add_argument('--init_gain', type=float, default=0.02, help='scaling factor for normal, xavier and orthogonal.')
        parser.add_argument('--no_dropout', action='store_true', help='no dropout for the generator')
        parser.add_argument('--dataset_mode', type=str, default='unaligned', help='chooses how datasets are loaded')
        parser.add_argument('--direction', type=str, default='AtoB', help='AtoB or BtoA')
        parser.add_argument('--serial_batches', action='store_true', help='if true, takes images in order to make batches')
        parser.add_argument('--num_threads', default=4, type=int, help='# threads for loading data')
        parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
        parser.add_argument('--name', type=str, default='experiment_name', help='name of the experiment')
        parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        parser.add_argument('--display_dir', type=str, default='./checkpoints', help='path to save the html')
        
        parser.add_argument('--model', type=str, default='cycle_gan', help='chooses which model to use')
        parser.add_argument('--epoch', type=str, default='latest', help='which epoch to load?')
        parser.add_argument('--load_iter', type=int, default='0', help='which iteration to load?')
        parser.add_argument('--verbose', action='store_true', help='if specified, print more debugging information')
        parser.add_argument('--suffix', default='', type=str, help='customized suffix')
        # =========================================================
        # [Exp 30 新增参数] 放在 BaseOptions 以确保测试时也能用
        # =========================================================
        parser.add_argument('--attn_temp', type=float, default=1.0, help='temperature for attention sharpening (default: 1.0)')
        parser.add_argument('--use_dilation', action='store_true', help='if specified, use dilated convolutions in the bottleneck')
        # 找到 parser.add_argument('--use_dilation', ...) 这一行，在它下面添加：

        parser.add_argument('--use_aspp', action='store_true', help='use ASPP module in the innermost layer of generator')
        # parser.add_argument('--lambda_bg', type=float, default=0.0, help='weight for background suppression loss')
        # parser.add_argument('--lambda_ffl', type=float, default=0.0, help='weight for focal frequency loss (e.g., 0.1)')
        # =========================================================
        # 3D 相关参数
        parser.add_argument('--patch_size_d', type=int, default=64, help='crop size d')
        parser.add_argument('--patch_size_h', type=int, default=128, help='crop size h')
        parser.add_argument('--patch_size_w', type=int, default=128, help='crop size w')
        parser.add_argument('--spacing_z', type=float, default=1.0, help='spacing z')
        parser.add_argument('--spacing_x', type=float, default=1.0, help='spacing x')
        parser.add_argument('--norm_min', type=float, default=-60.0, help='dB min')
        parser.add_argument('--norm_max', type=float, default=0.0, help='dB max')
        parser.add_argument('--no_elastic', action='store_true', help='disable elastic transform')
        parser.add_argument('--no_flip', action='store_true', help='disable flip')
        parser.add_argument('--upsample_mode', type=str, default='trilinear', help='upsampling mode in generator: trilinear | nearest')

        self.initialized = True
        return parser

    def gather_options(self):
        if not self.initialized:
            parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)

        # 获取基本参数
        opt, _ = parser.parse_known_args()

        # 修改 model 相关的 parser 选项
        model_name = opt.model
        model_option_setter = models.get_option_setter(model_name)
        parser = model_option_setter(parser, self.isTrain)
        opt, _ = parser.parse_known_args()

        # 修改 dataset 相关的 parser 选项
        dataset_name = opt.dataset_mode
        dataset_option_setter = data.get_option_setter(dataset_name)
        parser = dataset_option_setter(parser, self.isTrain)

        self.parser = parser
        return parser.parse_args()

    def print_options(self, opt):
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)

        if hasattr(opt, 'checkpoints_dir') and hasattr(opt, 'name'):
            expr_dir = os.path.join(opt.checkpoints_dir, opt.name)
            util.mkdirs(expr_dir)
            file_name = os.path.join(expr_dir, '{}_opt.txt'.format(opt.phase))
            with open(file_name, 'wt') as opt_file:
                opt_file.write(message)
                opt_file.write('\n')

    def parse(self):
        opt = self.gather_options()
        opt.isTrain = self.isTrain
        if opt.suffix:
            suffix = ('_' + opt.suffix.format(**vars(opt))) if opt.suffix != '' else ''
            opt.name = opt.name + suffix
        self.print_options(opt)
        str_ids = opt.gpu_ids.split(',')
        opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                opt.gpu_ids.append(id)
        if len(opt.gpu_ids) > 0:
            torch.cuda.set_device(opt.gpu_ids[0])
        self.opt = opt
        return self.opt