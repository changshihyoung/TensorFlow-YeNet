import os
import shutil
import numpy as np
import tensorflow as tf
from scipy import misc, io
from random import shuffle

def get_files(cover_dir, stego_dir, use_shuf_pair=False):
    """
    从cover和stego文件夹中提取图片，返回到get_batches组成batch
    shuf_pair决定了组成batch时，cover与stego是否成对
    """
    file = []
    for filename in os.listdir(cover_dir + '/'):
        file.append(filename)
    shuffle(file)
    file_shuf1 = file

    img = []
    img_label = []
    if use_shuf_pair:
        shuffle(file)
        file_shuf2 = file
        for file_idx in range(len(file_shuf1)):
            img.append(cover_dir + '/' + file_shuf1[file_idx])
            img_label.append(0)
            img.append(stego_dir + '/' + file_shuf2[file_idx])
            img_label.append(1)
    else:
        for filename in file_shuf1:
            img.append(cover_dir + '/' + filename)
            img_label.append(0)
            img.append(stego_dir + '/' + filename)
            img_label.append(1)

    #将img_list和img_label写入cover路径下的img_label_list.txt
    #with open(cover_dir + '/' + 'img_label_list.txt', 'w') as f:
    #    for img_idx in range(len(img)):
    #        f.write(img[img_idx]+' '+str(img_label[img_idx])+'\n')
            
    return img, img_label

def get_minibatches(img, img_label, batch_size):
    """
    替代get_batches函数的作用，批次读取数据，每次返回batch_size大小的数据
    """
    for start_idx in range(0, len(img) - batch_size + 1, batch_size):
        excerpt = slice(start_idx, start_idx + batch_size)
        img_minibatch = img[excerpt]
        img_label_minibatch = img_label[excerpt]
        yield img_minibatch, img_label_minibatch

def get_minibatches_content_img(train_img_minibatch_list, img_height, img_width):
    """
    读取get_minibatches函数返回路径对应的内容，将图片实际内容转换为batch，作为返回值
    """
    img_num = len(train_img_minibatch_list)
    image_minibatch_content = np.zeros([img_num, img_height, img_width, 1], dtype=np.float32)

    i = 0
    for img_file in train_img_minibatch_list:
        content = misc.imread(img_file)
        image_minibatch_content[i, :, :, 0] = content
        i = i + 1

    return image_minibatch_content

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

"""
def get_batches(img, img_label, batch_size, capacity):
    #
    #根据get_files返回的图片列表和标签列表，生成训练用batch
    #需要注意的是：输入图片应具有相同的高、宽
    #
    img = tf.cast(img, tf.string)
    img_label = tf.cast(img_label, tf.int32)

    # 生成输入队列（queue），tensorflow有多种方法，这里展示image与label分开时的情况
    input_queue = tf.train.slice_input_producer([img, img_label])

    # 从队列里读出label，image（需要对相应的图片进行解码）
    label = input_queue[1]
    image_contents = tf.read_file(input_queue[0])  #pgm图像不能这么使用
    image = tf.image.decode_image(image_contents, channels=1)  #pgm图像不能这么使用
    ##数据集augmentation的部分

    # 对数据进行大小标准化等操作，tf.image下有很多对image的处理，randomflip等
    #image = tf.image.resize_image_with_crop_or_pad(image, image_W, image_H)
    #image = tf.image.per_image_standardization(image)

    #[image, label]是tensor型变量
    image_batch, label_batch = tf.train.batch([image, label],
                                              batch_size=batch_size,
                                              num_threads=64,
                                              capacity=capacity)
    label_batch = tf.reshape(label_batch, [batch_size])

    return image_batch, label_batch
"""