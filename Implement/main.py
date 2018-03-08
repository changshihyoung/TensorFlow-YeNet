import argparse
import numpy as np
import tensorflow as tf
from glob import glob
from utils import *
from generator import *
from YeNet import YeNet

# *定义命令行输入变量
parser = argparse.ArgumentParser(description='Tensorflow implementation of YeNet')

# *根据不同操作进行不同的命令行参数定义
input_operation = input('The operation you want to perform(train, test, datasplit):')
if 'train' in input_operation:
    parser.add_argument('train_cover_dir', type=str, metavar='PATH',
                        help='directory of training cover images')
    parser.add_argument('train_stego_dir', type=str, metavar='PATH',
                        help='directory of training stego images')
    parser.add_argument('valid_cover_dir', type=str, metavar='PATH',
                        help='directory of validation cover images')
    parser.add_argument('valid_stego_dir', type=str, metavar='PATH',
                        help='directory of validation stego images')
if 'test' in input_operation:
    parser.add_argument('test_cover_dir', type=str, metavar='PATH',
                        help='directory of testing cover images')
    parser.add_argument('test_stego_dir', type=str, metavar='PATH',
                        help='directory of testing stego images')
if 'datasplit' in input_operation:
    parser.add_argument('source_dir', type=str, metavar='PATH',
                        help='directory of source cover and stego images')
    parser.add_argument('dest_dir', type=str, metavar='PATH',
                        help='directory of separated dataset ')
if 'train' not in input_operation\
        and 'test' not in input_operation \
        and 'datasplit' not in input_operation:
    raise NotImplementedError('invalid operation')

# *定义余下可选命令行参数
parser.add_argument('--use-shuf-pair', action='store_true', default=False,
                    help='matching cover and stego image when batch is constructed (default: False)')
parser.add_argument('--use-batch-norm', action='store_true', default=False,
                    help='use batch normalization after each activation (default: False)')
parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                    help='input batch size for training, testing and validation (default: 32)')
parser.add_argument('--max-epochs', type=int, default=200, metavar='N',
                    help='number of epochs to train (default: 200)')
parser.add_argument('--lr', type=float, default=4e-1, metavar='F',
                    help='learning rate (default: 4e-1)')
parser.add_argument('--gpu', type=str, default='0', metavar='S',
                    help='index of gpu used (default: 0)')
parser.add_argument('--tfseed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=20, metavar='N',
                    help='number of batches before logging training status')
parser.add_argument('--log-path', type=str, default='logs/',
                    metavar='PATH', help='directory of log file')
parser.add_argument('--train-percent', type=float, default=0.6, metavar='F',
                    help='percentage of dataset used for training (default: 0.6)')
parser.add_argument('--valid-percent', type=float, default=0.2, metavar='F',
                    help='percentage of dataset used for validation (default: 0.2)')
parser.add_argument('--test-percent', type=float, default=0.2, metavar='F',
                    help='percentage of dataset used for testing (default: 0.2)')
args = parser.parse_args()

import os
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

# *设置tf随机种子
tf.set_random_seed(args.tfseed)

# *根据不同操作输入执行相应函数
if 'datasplit' in input_operation:
    # *数据集分割主函数
    data_split(args.source_dir, args.dest_dir,
               args.batch_size,
               train_percent=args.train_percent,
               valid_percent=args.valid_percent,
               test_percent=args.test_percent)

if 'train' in input_operation:
    # *计算train/valid数据集大小
    train_ds_size = len(glob(args.train_cover_dir + '/*')) * 2
    if train_ds_size % args.batch_size != 0:
        raise ValueError('change batch size for training')
    valid_ds_size = len(glob(args.valid_cover_dir + '/*')) * 2
    if valid_ds_size % args.batch_size != 0:
        raise ValueError('change batch size for validation')
    # *训练主函数
    train(YeNet, args.use_batch_norm, args.use_shuf_pair,
          args.train_cover_dir, args.train_stego_dir,
          args.valid_cover_dir, args.valid_stego_dir,
          args.batch_size, train_ds_size, valid_ds_size,
          args.log_interval, args.max_epochs, args.lr,
          args.log_path)

if 'test' in input_operation:
    # *计算test数据集大小
    test_ds_size = len(glob(args.test_cover_dir + '/*')) * 2
    if test_ds_size % args.batch_size != 0:
        raise ValueError('change batch size for testing')
    # *查找最佳模型主函数
    test_dataset_findbest(YeNet, args.use_shuf_pair,
                          args.test_cover_dir, args.test_stego_dir, args.max_epochs,
                          args.batch_size, test_ds_size, args.log_path)


