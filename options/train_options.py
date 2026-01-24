# from .base_options import BaseOptions

# class TrainOptions(BaseOptions):
#     """This class includes training options."""

#     def initialize(self, parser):
#         parser = BaseOptions.initialize(self, parser)
        
#         # === [核心修改] 把 Loss 权重全搬到这里 ===
#         parser.add_argument('--lambda_gan', type=float, default=1.0, help='weight for GAN loss')
#         parser.add_argument('--lambda_pixel', type=float, default=100.0, help='weight for pixel loss (L1)')
#         parser.add_argument('--lambda_perceptual', type=float, default=0.0, help='weight for perceptual loss (VGG)')
#         parser.add_argument('--lambda_tv', type=float, default=0.0, help='weight for TV loss')
#         parser.add_argument('--lambda_edge', type=float, default=0.0, help='weight for edge loss')
#         # =======================================

#         # 保持原有的其他参数
#         parser.add_argument('--display_freq', type=int, default=400, help='frequency of showing training results on screen')
#         parser.add_argument('--print_freq', type=int, default=100, help='frequency of showing training results on console')
#         parser.add_argument('--save_latest_freq', type=int, default=5000, help='frequency of saving the latest results')
#         parser.add_argument('--save_epoch_freq', type=int, default=5, help='frequency of saving checkpoints at the end of epochs')
#         parser.add_argument('--save_by_iter', action='store_true', help='whether saves model by iteration')
#         parser.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')
#         parser.add_argument('--epoch_count', type=int, default=1, help='the starting epoch count')
#         parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')
#         parser.add_argument('--lr_d_ratio', type=float, default=1.0, help='learning rate ratio for D relative to G')
        
#         # 学习率相关
#         parser.add_argument('--n_epochs', type=int, default=100, help='number of epochs with the initial learning rate')
#         parser.add_argument('--n_epochs_decay', type=int, default=100, help='number of epochs to linearly decay learning rate to zero')
#         parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
#         parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate for adam')
#         parser.add_argument('--gan_mode', type=str, default='lsgan', help='the type of GAN objective. [vanilla| lsgan | wgangp]')
#         parser.add_argument('--pool_size', type=int, default=50, help='the size of image buffer that stores previously generated images')
#         parser.add_argument('--lr_policy', type=str, default='linear', help='learning rate policy. [linear | step | plateau | cosine]')
#         parser.add_argument('--lr_decay_iters', type=int, default=50, help='multiply by a gamma every lr_decay_iters iterations')

#         self.isTrain = True
#         return parser

import argparse
import os
from util import util
import torch
import models
import data

class BaseOptions():
    """This class defines options used during both training and test time."""

    def initialize(self, parser):
        # ... (保留原有参数) ...
        parser.add_argument('--dataroot', required=True, help='path to images (should have subfolders for train/test)')
        parser.add_argument('--batch_size', type=int, default=1, help='input batch size')
        parser.add_argument('--loadSize', type=int, default=286, help='scale images to this size')
        parser.add_argument('--cropSize', type=int, default=256, help='then crop to this size')
        parser.add_argument('--input_nc', type=int, default=1, help='# of input image channels: 3 for RGB and 1 for grayscale')
        parser.add_argument('--output_nc', type=int, default=1, help='# of output image channels: 3 for RGB and 1 for grayscale')
        
        # === [新增] 数据文件夹名称控制 ===
        parser.add_argument('--dir_lq', type=str, default='Recon_LQ_03', help='folder name for input data (LQ)')
        parser.add_argument('--dir_sq', type=str, default='Recon_SQ_75', help='folder name for target data (SQ)')
        # ==============================

        parser.add_argument('--ngf', type=int, default=64, help='# of gen filters in the last conv layer')
        parser.add_argument('--ndf', type=int, default=64, help='# of discrim filters in the first conv layer')
        parser.add_argument('--netD', type=str, default='basic', help='specify discriminator architecture [basic | n_layers | pixel]')
        parser.add_argument('--netG', type=str, default='resnet_9blocks', help='specify generator architecture [resnet_9blocks | resnet_6blocks | unet_256 | unet_128]')
        parser.add_argument('--n_layers_D', type=int, default=3, help='only used if netD==n_layers')
        parser.add_argument('--norm', type=str, default='instance', help='instance normalization or batch normalization [instance | batch | none]')
        parser.add_argument('--init_type', type=str, default='normal', help='network initialization [normal | xavier | kaiming | orthogonal]')
        parser.add_argument('--init_gain', type=float, default=0.02, help='scaling factor for normal, xavier and orthogonal.')
        parser.add_argument('--no_dropout', action='store_true', help='no dropout for the generator')
        parser.add_argument('--dataset_mode', type=str, default='unaligned', help='chooses how datasets are loaded. [unaligned | aligned | single | colorization]')
        parser.add_argument('--direction', type=str, default='AtoB', help='AtoB or BtoA')
        parser.add_argument('--serial_batches', action='store_true', help='if true, takes images in order to make batches, otherwise takes them randomly')
        parser.add_argument('--num_threads', default=4, type=int, help='# threads for loading data')
        parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
        parser.add_argument('--name', type=str, default='experiment_name', help='name of the experiment. It decides where to store samples and models')
        parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        parser.add_argument('--display_dir', type=str, default='./checkpoints', help='path to save the html')
        
        # 补全可能缺失的参数
        parser.add_argument('--model', type=str, default='cycle_gan', help='chooses which model to use. [cycle_gan | pix2pix | test | colorization]')
        parser.add_argument('--epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
        parser.add_argument('--load_iter', type=int, default='0', help='which iteration to load? if load_iter > 0, the code will load models by iter_[load_iter]; otherwise, the code will load models by [epoch]')
        parser.add_argument('--verbose', action='store_true', help='if specified, print more debugging information')
        parser.add_argument('--suffix', default='', type=str, help='customized suffix: opt.name = opt.name + suffix: e.g., {model}_{netG}_size{loadSize}')
        
        # 3D 相关参数保留
        parser.add_argument('--patch_size_d', type=int, default=64, help='crop size d')
        parser.add_argument('--patch_size_h', type=int, default=128, help='crop size h')
        parser.add_argument('--patch_size_w', type=int, default=128, help='crop size w')
        parser.add_argument('--spacing_z', type=float, default=1.0, help='spacing z')
        parser.add_argument('--spacing_x', type=float, default=1.0, help='spacing x')
        parser.add_argument('--norm_min', type=float, default=-60.0, help='dB min')
        parser.add_argument('--norm_max', type=float, default=0.0, help='dB max')
        parser.add_argument('--no_elastic', action='store_true', help='disable elastic transform')
        parser.add_argument('--no_flip', action='store_true', help='disable flip')

        self.initialized = True
        return parser

    def gather_options(self):
        # Initialize
        if not self.initialized:
            parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)

        # get the basic options
        opt, _ = parser.parse_known_args()

        # modify model-related parser options
        model_name = opt.model
        model_option_setter = models.get_option_setter(model_name)
        parser = model_option_setter(parser, self.isTrain)
        opt, _ = parser.parse_known_args()  # parse again with new defaults

        # modify dataset-related parser options
        dataset_name = opt.dataset_mode
        dataset_option_setter = data.get_option_setter(dataset_name)
        parser = dataset_option_setter(parser, self.isTrain)

        # save and return the parser
        self.parser = parser
        return parser.parse_args()

    def print_options(self, opt):
        # 这里就是控制打印的代码，我们不需要大改，只需要确保 train.py 调用它
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

        # save to the disk
        if hasattr(opt, 'checkpoints_dir') and hasattr(opt, 'name'):
            expr_dir = os.path.join(opt.checkpoints_dir, opt.name)
            util.mkdirs(expr_dir)
            file_name = os.path.join(expr_dir, '{}_opt.txt'.format(opt.phase))
            with open(file_name, 'wt') as opt_file:
                opt_file.write(message)
                opt_file.write('\n')

    def parse(self):
        opt = self.gather_options()
        opt.isTrain = self.isTrain   # train or test

        # process opt.suffix
        if opt.suffix:
            suffix = ('_' + opt.suffix.format(**vars(opt))) if opt.suffix != '' else ''
            opt.name = opt.name + suffix

        self.print_options(opt)

        # set gpu ids
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