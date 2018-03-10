import os
import numpy as np
from scipy import misc
from glob import glob
import shutil
import random
from random import random as rand
from random import shuffle

# *数据集预处理主函数
def data_transfer(source_dir, dest_dir,
                  required_size,
                  required_operation):
    """
    将source_dir中的图像依operation中定义的操作扩展至dest_dir中
    包含resize，subsample和crop操作
    """
    dest_dir = dest_dir + '/' + source_dir.split("/")[-1]

    # 建立数据集路径
    size = required_size, required_size
    if 'resize' in required_operation:
        dest_resize_dir = dest_dir + '_' + str(required_size) + '_resize'
        if os.path.exists(dest_resize_dir + '/') is False:
            os.mkdir(dest_resize_dir + '/')
        os.mkdir(dest_resize_dir + '/cover/')
    if 'crop' in required_operation:
        dest_crop_dir = dest_dir + '_' + str(required_size) + '_crop'
        if os.path.exists(dest_crop_dir + '/') is False:
            os.mkdir(dest_crop_dir + '/')
        os.mkdir(dest_crop_dir + '/cover/')
    if 'subsample' in required_operation:
        dest_subsample_dir = dest_dir + '_' + str(required_size) + '_subsample'
        if os.path.exists(dest_subsample_dir + '/') is False:
            os.mkdir(dest_subsample_dir + '/')
        os.mkdir(dest_subsample_dir + '/cover/')

    source_img_list = glob(source_dir + '/*')
    for filename in source_img_list:
        img = misc.imread(filename)
        if img is None:
            raise OSError('Error: could not load image')
        if 'resize' in required_operation:
            img_resize = misc.imresize(img, size, interp='bicubic')
            save_dir = dest_resize_dir + '/cover/' + filename.split("/")[-1]
            misc.imsave(save_dir, img_resize)
        if 'crop' in required_operation:
            ROI_idx = (img.shape[0] - required_size) // 2
            img_crop = img[ROI_idx:ROI_idx+required_size, ROI_idx:ROI_idx+required_size]
            save_dir = dest_crop_dir + '/cover/' + filename.split("/")[-1]
            misc.imsave(save_dir, img_crop)
        if 'subsample' in required_operation:
            SUB_idx = img.shape[0] // required_size
            img_subsample = img[0:img.shape[0]:SUB_idx, 0:img.shape[1]:SUB_idx]
            save_dir = dest_subsample_dir + '/cover/' + filename.split("/")[-1]
            misc.imsave(save_dir, img_subsample)
    print('data transfer succeed!')

# *数据集增广主函数
def data_aug(source_dir, dest_dir, ratio=0.5):
    """
    将source_dir中的图像增广至dest_dir中
    包含rotate和flip操作
    """
    dest_dir = dest_dir + '/' + source_dir.split("/")[-2] + '_aug'
    if os.path.exists(dest_dir + '/') is False:
        os.mkdir(dest_dir + '/')
    os.mkdir(dest_dir + '/cover/')

    dest_dir = dest_dir + '/cover'

    source_img_list = glob(source_dir + '/*')
    for filename in source_img_list:
        img = misc.imread(filename)
        if img is None:
            raise OSError('Error: could not load image')
        filename_split = (filename.split("/")[-1])
        save_dir = dest_dir + '/' + filename_split
        misc.imsave(save_dir, img)

        rot = random.randint(1, 3)
        rand_op = rand()
        rand_flip = rand()
        if rand_op < ratio:
            img_rot = misc.imrotate(img, rot*90, interp='bicubic')
            save_dir = dest_dir + '/' + filename_split.split('.')[0] + '_rot.' + filename_split.split('.')[1]
            misc.imsave(save_dir, img_rot)
        else:
            if rand_flip < ratio:
                img_flip = np.flipud(img)
            else:
                img_flip = np.fliplr(img)
            save_dir = dest_dir + '/' + filename_split.split('.')[0] + '_flip.' + filename_split.split('.')[1]
            misc.imsave(save_dir, img_flip)
    print('data augment succeed!')


# *数据集分割主函数
def data_split(source_dir, dest_dir,
               batch_size,
               train_percent=0.6,
               valid_percent=0.2,
               test_percent=0.2):
    """
    根据传入的source_dir中cover/stego图像路径，根据各percent参数
    在dest_dir路径中分割成train/valid/test数据集
    抽取方式是随机的
    """
    # *判断输入百分比是否合法
    if (train_percent + valid_percent + test_percent) > 1:
        raise ValueError('sum of train valid test percentage larger than 1')

    if os.path.exists(dest_dir + '/') is False:
        os.mkdir(dest_dir + '/')
    if os.path.exists(source_dir + '/') is False:
        raise OSError('source direction not exist')

    source_cover_dir = source_dir + '/cover'
    source_stego_dir = source_dir + '/stego'

    # *清理非对应文件
    file_clean(source_cover_dir, source_stego_dir)

    # *在dest_dir路径下创建train/valid/test路径
    dest_train_dir, dest_valid_dir, dest_test_dir = file_dir_mk_trainvalidtest_dir(dest_dir)

    # *对source_dir中的文件顺序进行shuffle
    source_cover_list = []
    for filename in os.listdir(source_cover_dir + '/'):
        source_cover_list.append(filename)
    shuffle(source_cover_list)

    # *计算train/valid/test数据集容量
    half_batch_size = batch_size // 2
    train_ds_capacity = ( int( len(source_cover_list)*train_percent ) // half_batch_size ) * half_batch_size
    valid_ds_capacity = ( int( len(source_cover_list)*valid_percent ) // half_batch_size ) * half_batch_size
    test_ds_capacity  = ( int( len(source_cover_list)*test_percent  ) // half_batch_size ) * half_batch_size

    for fileidx in range(train_ds_capacity):
        srcfile_cover = source_cover_dir + '/' + source_cover_list[fileidx]
        dstfile_cover = dest_train_dir + '/cover/' + source_cover_list[fileidx]
        shutil.copyfile(srcfile_cover, dstfile_cover)
        srcfile_stego = source_stego_dir + '/' + source_cover_list[fileidx]
        dstfile_stego = dest_train_dir + '/stego/' + source_cover_list[fileidx]
        shutil.copyfile(srcfile_stego, dstfile_stego)
    for fileidx in range(train_ds_capacity, train_ds_capacity + valid_ds_capacity):
        srcfile_cover = source_cover_dir + '/' + source_cover_list[fileidx]
        dstfile_cover = dest_valid_dir + '/cover/' + source_cover_list[fileidx]
        shutil.copyfile(srcfile_cover, dstfile_cover)
        srcfile_stego = source_stego_dir + '/' + source_cover_list[fileidx]
        dstfile_stego = dest_valid_dir + '/stego/' + source_cover_list[fileidx]
        shutil.copyfile(srcfile_stego, dstfile_stego)
    for fileidx in range(train_ds_capacity + valid_ds_capacity,
                          train_ds_capacity + valid_ds_capacity + test_ds_capacity):
        srcfile_cover = source_cover_dir + '/' + source_cover_list[fileidx]
        dstfile_cover = dest_test_dir + '/cover/' + source_cover_list[fileidx]
        shutil.copyfile(srcfile_cover, dstfile_cover)
        srcfile_stego = source_stego_dir + '/' + source_cover_list[fileidx]
        dstfile_stego = dest_test_dir + '/stego/' + source_cover_list[fileidx]
        shutil.copyfile(srcfile_stego, dstfile_stego)
    print('data split succeed!')

def file_clean(cover_dir, stego_dir):
    """
    对cover和stego里的文件进行清理，将只存在于单个文件夹的文件、后缀名不匹配的文件删除。
    """
    cover_dir = cover_dir + '/'
    stego_dir = stego_dir + '/'
    cover_list = []
    stego_list = []
    for root, dirs, files in os.walk(cover_dir):
        for filenames in files:
            cover_list.append(filenames)
    for root, dirs, files in os.walk(stego_dir):
        for filenames in files:
            stego_list.append(filenames)
    diff_cover_list = set(cover_list).difference(set(stego_list))
    diff_stego_list = set(stego_list).difference(set(cover_list))
    print('Start file cleaning...')
    print('About to delete: ', len(diff_cover_list), 'files in ', cover_dir)
    for filenames in diff_cover_list:
        os.remove(cover_dir + filenames)
    print('About to delete: ', len(diff_stego_list), 'files in ', stego_dir)
    for filenames in diff_stego_list:
        os.remove(stego_dir + filenames)

def file_dir_mk_trainvalidtest_dir(dest_dir):
    """
    在dest_dir路径下创建train/valid/test路径
    """
    if os.path.exists(dest_dir + '/train/') is False:
        os.mkdir(dest_dir + '/train/')
    if os.path.exists(dest_dir + '/train/cover/') is False:
        os.mkdir(dest_dir + '/train/cover/')
    if os.path.exists(dest_dir + '/train/stego/') is False:
        os.mkdir(dest_dir + '/train/stego/')
    if os.path.exists(dest_dir + '/valid/') is False:
        os.mkdir(dest_dir + '/valid/')
    if os.path.exists(dest_dir + '/valid/cover/') is False:
        os.mkdir(dest_dir + '/valid/cover/')
    if os.path.exists(dest_dir + '/valid/stego/') is False:
        os.mkdir(dest_dir + '/valid/stego/')
    if os.path.exists(dest_dir + '/test/') is False:
        os.mkdir(dest_dir + '/test/')
    if os.path.exists(dest_dir + '/test/cover/') is False:
        os.mkdir(dest_dir + '/test/cover/')
    if os.path.exists(dest_dir + '/test/stego/') is False:
        os.mkdir(dest_dir + '/test/stego/')
    if os.path.exists(dest_dir + '/log/') is False:
        os.mkdir(dest_dir + '/log/')
    return dest_dir + '/train', dest_dir + '/valid', dest_dir + '/test'
