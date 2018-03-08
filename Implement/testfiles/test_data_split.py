import os
import shutil
from random import shuffle

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
    print('About to delete: ', len(diff_cover_list), 'files in ', cover_dir, 'Continue?')
    os.system('pause')
    for filenames in diff_cover_list:
        os.remove(cover_dir + filenames)
    print('About to delete: ', len(diff_stego_list), 'files in ', stego_dir, 'Continue?')
    os.system('pause')
    for filenames in diff_stego_list:
        os.remove(stego_dir + filenames)
    print('file_clean process has completed.')

if __name__ == '__main__':
    source_dir = 'E:\@ChangShihyoung\TensorFlow-YeNet\Data\SUNI_13_0.4'
    dest_dir = 'E:\@ChangShihyoung\TensorFlow-YeNet\Experiment\SUNI_13_0.4_No_1'
    data_split(source_dir, dest_dir,
               4,
               train_percent=0.6,
               valid_percent=0.2,
               test_percent=0.2)