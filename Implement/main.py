import argparse
import numpy as np
import tensorflow as tf
from glob import glob

from utils import *
from generator import *
from YeNet import YeNet
##分成main_train.py和main_test.py?##缺少learning rate decay##
parser = argparse.ArgumentParser(description='Tensorflow implementation of YeNet')
parser.add_argument('train_cover_dir', type=str, metavar='PATH',
                    help='path of directory containing all ' +
                    'training cover images')
parser.add_argument('train_stego_dir', type=str, metavar='PATH',
                    help='path of directory containing all ' +
                    'training stego images or beta maps')
parser.add_argument('valid_cover_dir', type=str, metavar='PATH',
                    help='path of directory containing all ' +
                    'validation cover images')
parser.add_argument('valid_stego_dir', type=str, metavar='PATH',
                    help='path of directory containing all ' +
                    'validation stego images or beta maps')
parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                    help='input batch size for training and validation (default: 32)')
parser.add_argument('--test-batch-size', type=int, default=32, metavar='N',
                    help='input batch size for testing (default: 32)')
parser.add_argument('--max-epochs', type=int, default=200, metavar='N',
                    help='number of epochs to train (default: 200)')
parser.add_argument('--lr', type=float, default=4e-1, metavar='LR',
                    help='learning rate (default: 4e-1)')
parser.add_argument('--use-shuf-pair', action='store_true', default=False,
                    help='matching cover and stego when batch is constructed' +
                    ' (default: False)')
parser.add_argument('--use-batch-norm', action='store_true', default=False,
                    help='use batch normalization after each activation,' +
                    ' also disable pair constraint (default: False)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--gpu', type=str, default='0', metavar='S',
                    help='index of gpu used (default: 0)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=20, metavar='N',
                    help='how many batches to wait ' +
                    'before logging training status')
parser.add_argument('--log-path', type=str, default='logs/',
                    metavar='PATH', help='path to generated log file')
args = parser.parse_args()

import os
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '' if args.no_cuda else args.gpu

#设置tf随机种子
tf.set_random_seed(args.seed)

#清理非对应文件
#file_clean(args.train_cover_dir, args.train_stego_dir)
#file_clean(args.valid_cover_dir, args.valid_stego_dir)

#计算数据集大小
train_ds_size = len(glob(args.train_cover_dir + '/*')) * 2
if train_ds_size % args.batch_size != 0:
    raise ValueError("change batch size for training")
valid_ds_size = len(glob(args.valid_cover_dir + '/*')) * 2
if valid_ds_size % args.batch_size != 0:
    raise ValueError("change batch size for validation")

#optimizer部分可以调整
optimizer = tf.train.AdadeltaOptimizer(args.lr)

##训练主函数
train(YeNet, args.use_batch_norm, args.use_shuf_pair,
      args.train_cover_dir, args.train_stego_dir,
      args.valid_cover_dir, args.valid_stego_dir,
      args.batch_size, train_ds_size, valid_ds_size,
      optimizer, args.log_interval, args.max_epochs,
      args.log_path)

##查找最佳模型主函数