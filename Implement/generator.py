import os
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