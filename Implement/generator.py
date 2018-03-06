import os
import numpy as np
import tensorflow as tf
from scipy import misc, io
from glob import glob
import random
from random import random as rand
from random import shuffle

def file_clean(cover_dir, stego_dir):
    '''
    对cover和stego里的文件进行清理，将只存在于单个文件夹的文件、后缀名不匹配的文件删除。
    '''
    cover_dir = cover_dir + '/'
    stego_dir = stego_dir + '/'
    cover_list=[]
    stego_list=[]
    for root,dirs,files in os.walk(cover_dir):
        for filenames in files:
            cover_list.append(filenames)
    for root,dirs,files in os.walk(stego_dir):
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

def get_files(cover_dir, stego_dir, use_shuf_pair=False):
    '''
    从cover和stego文件夹中提取图片，返回到get_batches组成batch
    shuf_pair决定了组成batch时，cover与stego是否成对
    '''
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
    '''
    替代get_batches函数的作用，批次读取数据，每次返回batch_size大小的数据
    '''
    for start_idx in range(0, len(img) - batch_size + 1, batch_size):
        excerpt = slice(start_idx, start_idx + batch_size)
        img_minibatch = img[excerpt]
        img_label_minibatch = img_label[excerpt]
        yield img_minibatch, img_label_minibatch

def get_minibatches_content_img(train_img_minibatch_list, img_height, img_width):
    '''
    读取get_minibatches函数返回路径对应的内容，将图片实际内容转换为batch，作为返回值
    '''
    img_num = len(train_img_minibatch_list)
    image_minibatch_content = np.zeros([img_num, img_height, img_width, 1], dtype=np.float32)

    i = 0
    for img_file in train_img_minibatch_list:
        content = misc.imread(img_file)
        image_minibatch_content[i,:,:,0] = content
        i = i + 1

    return image_minibatch_content

'''
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
'''

#-----------------------------------------------

def gen_all_flip_and_rot(cover_dir, stego_dir, thread_idx, n_threads):
    cover_list = sorted(glob(cover_dir + '/*'))
    stego_list = sorted(glob(stego_dir + '/*'))
    nb_data = len(cover_list)
    assert len(stego_list) != 0, "the beta directory '%s' is empty" % stego_dir
    assert nb_data != 0, "the cover directory '%s' is empty" % cover_dir
    assert len(stego_list) == nb_data, "the cover directory and " + \
                                      "the beta directory don't " + \
                                      "have the same number of files " + \
                                      "respectively %d and %d" % (nb_data, + \
                                      len(stego_list))
    img = misc.imread(cover_list[0])
    img_shape = img.shape
    batch = np.empty((2,img_shape[0],img_shape[1],1), dtype='uint8')
    iterable = zip(cover_list, stego_list)
    for cover_path, stego_path in iterable:
        batch[0,:,:,0] = misc.imread(cover_path)
        batch[1,:,:,0] = misc.imread(stego_path)
        for rot in range(4):
            yield [np.rot90(batch, rot, axes=[1,2]), np.array([0,1], dtype='uint8')]
        for rot in range(4):
            yield [np.flip(np.rot90(batch, rot, axes=[1,2]), axis=2), np.array([0,1], dtype='uint8')]

def gen_flip_and_rot(cover_dir, stego_dir, shuf_pair=False, thread_idx=0, n_threads=1):
    cover_list = sorted(glob(cover_dir + '/*'))
    stego_list = sorted(glob(stego_dir + '/*'))
    nb_data = len(cover_list)
    assert len(stego_list) != 0, "the beta directory '%s' is empty" % stego_dir
    assert nb_data != 0, "the cover directory '%s' is empty" % cover_dir
    assert len(stego_list) == nb_data, "the cover directory and " + \
                                      "the beta directory don't " + \
                                      "have the same number of files " + \
                                      "respectively %d and %d" % (nb_data, + \
                                      len(stego_list))
    img = misc.imread(cover_list[0])
    img_shape = img.shape
    batch = np.empty((2,img_shape[0],img_shape[1],1), dtype='uint8')
    if not shuf_pair:##zsy改动：zip在python3中返回的是iterator，无法shuffle()
        iterable = zip(cover_list, stego_list)
    while True:
        if shuf_pair:
            shuffle(cover_list)
            shuffle(stego_list)
            iterable = zip(cover_list, stego_list)
        #else:
            #shuffle(iterable)  ##zsy改动：把shuffle的过程去掉
        for cover_path, stego_path in iterable:
            batch[0,:,:,0] = misc.imread(cover_path)
            batch[1,:,:,0] = misc.imread(stego_path)
            rot = random.randint(0,3)
            if rand() < 0.5:
                yield [np.rot90(batch, rot, axes=[1,2]), np.array([0,1], dtype='uint8')]
            else:
                yield [np.flip(np.rot90(batch, rot, axes=[1,2]), axis=2), np.array([0,1], dtype='uint8')]

def gen_valid(cover_dir, stego_dir, thread_idx, n_threads):
    cover_list = sorted(glob(cover_dir + '/*'))
    stego_list = sorted(glob(stego_dir + '/*'))
    nb_data = len(cover_list)
    assert len(stego_list) != 0, "the beta directory '%s' is empty" % stego_dir
    assert nb_data != 0, "the cover directory '%s' is empty" % cover_dir
    assert len(stego_list) == nb_data, "the cover directory and " + \
                                      "the beta directory don't " + \
                                      "have the same number of files " + \
                                      "respectively %d and %d" % (nb_data, \
                                      len(stego_list))
    img = misc.imread(cover_list[0])
    img_shape = img.shape
    batch = np.empty((2,img_shape[0],img_shape[1],1), dtype='uint8')
    labels = np.array([0, 1], dtype='uint8')
    while True:
        for cover_path, stego_path in zip(cover_list, stego_list):
            batch[0,:,:,0] = misc.imread(cover_path)
            batch[1,:,:,0] = misc.imread(stego_path)
            yield [batch, labels]