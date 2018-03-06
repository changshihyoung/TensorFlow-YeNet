import tensorflow as tf
import numpy as np
from scipy import misc, io
import time
from glob import glob
from generator import *

class average_summary(object):
    def __init__(self, variable, name, num_iterations):
        self.sum_variable = tf.get_variable(name, shape=[],
                                initializer=tf.constant_initializer(0.),
                                dtype='float32',
                                trainable=False,
                                collections=[tf.GraphKeys.LOCAL_VARIABLES])
        with tf.control_dependencies([variable]):
            self.increment_op = tf.assign_add(self.sum_variable, variable)
        self.mean_variable = self.sum_variable / float(num_iterations)
        self.summary = tf.summary.scalar(name, self.mean_variable)
        with tf.control_dependencies([self.summary]):
            self.reset_variable_op = tf.assign(self.sum_variable, 0)

    def add_summary(self, sess, writer, step):
        s, _ = sess.run([self.summary, self.reset_variable_op])
        writer.add_summary(s, step)

##用于挂载Net的结构，包含__build_model和__build_loss的操作
class Model(object):
    def __init__(self, is_training=None, data_format='NCHW'):
        self.data_format = data_format
        if is_training is None:
            self.is_training = tf.get_variable('is_training', dtype=tf.bool,
                                    initializer=tf.constant_initializer(True),
                                    trainable=False)
        else:
            self.is_training = is_training

    def _build_model(self, inputs):
        raise NotImplementedError('Here is your model definition')

    def _build_losses(self, labels):
        self.labels = tf.cast(labels, tf.int64)
        with tf.variable_scope('loss'):
            oh = tf.one_hot(self.labels, 2)  #这里定义了2分类的输出
            ##除softmax cross entropy之外，还可更换其他函数
            self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                                       labels=oh, logits=self.outputs))
        with tf.variable_scope('accuracy'):
            am = tf.argmax(self.outputs, 1)
            equal = tf.equal(am, self.labels)
            self.accuracy = tf.reduce_mean(tf.cast(equal, tf.float32))
        return self.loss, self.accuracy

##训练主函数
def train(model_class, use_batch_norm, use_shuf_pair,
          train_cover_dir, train_stego_dir,
          valid_cover_dir, valid_stego_dir,
          batch_size, train_ds_size, valid_ds_size,
          optimizer, log_interval, max_epochs,
          log_path, load_path=None):
    ##清除默认图的堆栈，设置全局图为默认图
    tf.reset_default_graph()

    ##is_training用于判断训练处于train或者valid状态
    is_training = tf.get_variable('is_training', dtype=tf.bool,
                                  initializer=True, trainable=False)

    ##定义train_op操作和valid_op操作，将is_training和batch_size设置为对应的状态
    disable_training_op = tf.assign(is_training, False)
    enable_training_op = tf.assign(is_training, True)

    ##模型初始化
    #设置占位符
    temp_cover_list = glob(train_cover_dir + '/*')
    temp_img = misc.imread(temp_cover_list[0])
    temp_img_shape = temp_img.shape
    img_batch = tf.placeholder(tf.float32, [batch_size, temp_img_shape[0], temp_img_shape[1], 1], name='input_image_batch')
    label_batch = tf.placeholder(tf.int32, [batch_size, ], name="input_label_batch")
    #使用占位符初始化模型
    model = model_class(is_training, 'NCHW', with_bn=use_batch_norm, tlu_threshold=3)
    model._build_model(img_batch)
    loss, accuracy = model._build_losses(label_batch)

    ##设置需要最小化的loss函数
    regularization_losses = tf.get_collection(
                          tf.GraphKeys.REGULARIZATION_LOSSES)
    regularized_loss = tf.add_n([loss] + regularization_losses)
    #定义train中使用的基于loss/acc的类（运行次数：log_interval）
    train_loss_s = average_summary(loss, 'train_loss', log_interval)
    train_accuracy_s = average_summary(accuracy, 'train_accuracy', log_interval)
    #定义valid中使用的基于loss/acc的类（运行次数：valid_ds_size / valid_batch_size）
    valid_loss_s = average_summary(loss, 'valid_loss',
                                   float(valid_ds_size) / float(batch_size))
    valid_accuracy_s = average_summary(accuracy, 'valid_accuracy',
                                       float(valid_ds_size) / float(batch_size))

    ##全局变量global_step，从0开始进行计数
    global_step = tf.get_variable('global_step', dtype=tf.int32, shape=[],
                                  initializer=tf.constant_initializer(0),
                                  trainable=False)

    ##定义train及valid过程中需要用到的操作
    #核心操作：最小化loss
    minimize_op = optimizer.minimize(regularized_loss, global_step)
    #训练操作（每个iteration都要用）：最小化loss；train_loss累加；train_acc累加
    train_op = tf.group(minimize_op, train_loss_s.increment_op,
                        train_accuracy_s.increment_op)
    #验证操作（一个epoch结束后，每个valid中的iteration都要用）：valid_loss累加；valid_acc累加
    valid_op = tf.group(valid_loss_s.increment_op,
                        valid_accuracy_s.increment_op)
    #初始化操作：初始化所有的全局变量和局部变量
    init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())

    ##定义模型保存变量，最大存储max_to_keep个模型
    saver = tf.train.Saver(max_to_keep=max_epochs+20)
    global_valid_accuracy = 0  #全局valid_acc最大值

    ##会话开始
    with tf.Session() as sess:
        #初始化所有的全局变量和局部变量
        sess.run(init_op)
        #重载模型
        if load_path is not None:
            loader = tf.train.Saver(reshape=True)
            loader.restore(sess, load_path)
        #定义模型及参数保存位置
        writer = tf.summary.FileWriter(log_path + '/LogFile/', sess.graph)

        #初始化全局global_step
        start = sess.run(global_step)
        #初始化train/valid的loss和acc变量
        sess.run([valid_loss_s.reset_variable_op,
                  valid_accuracy_s.reset_variable_op,
                  train_loss_s.reset_variable_op,
                  train_accuracy_s.reset_variable_op])


        ##训练开始
        print('Start training...')
        global_train_batch = 0  #全局batch计数
        for epoch in range(max_epochs):
            start_time = time.time()
            train_img_list, train_label_list = get_files(train_cover_dir, train_stego_dir, use_shuf_pair=use_shuf_pair)
            valid_img_list, valid_label_list = get_files(valid_cover_dir, valid_stego_dir, use_shuf_pair=use_shuf_pair)

            ##训练开始：train
            sess.run(enable_training_op)  # 转换为train训练状态
            local_train_batch = 0  #局部batch计数
            for train_img_minibatch_list, train_label_minibatch_list in get_minibatches(train_img_list, train_label_list, batch_size):
                # minibatch数据读取
                train_img_batch = get_minibatches_content_img(train_img_minibatch_list, temp_img_shape[0], temp_img_shape[1])

                #train操作及指标显示
                sess.run(train_op, feed_dict={img_batch: train_img_batch, label_batch: train_label_minibatch_list})

                global_train_batch += 1
                local_train_batch += 1

                local_train_loss = train_loss_s.mean_variable
                local_train_accuracy = train_accuracy_s.mean_variable
                local_train_loss_value = local_train_loss.eval(session=sess)
                local_train_accuracy_value = local_train_accuracy.eval(session=sess)
                print('-TRAIN- epoch: %d batch: %d | train_loss: %f train_acc: %f' % (epoch, local_train_batch, local_train_loss_value, local_train_accuracy_value))

                #达到log_interval的标准后，对train_loss/acc进行存储
                if global_train_batch % log_interval == 0:
                    train_loss_s.add_summary(sess, writer, global_train_batch)
                    train_accuracy_s.add_summary(sess, writer, global_train_batch)

                #对最后20个模型进行存储
                if ((train_ds_size // batch_size) * max_epochs - global_train_batch) < 20:
                    saver.save(sess, log_path + '/Model_' + str(epoch) + '.ckpt')
                    print('---EPOCH:%d LAST:%d--- model has been saved' % (epoch, (train_ds_size // batch_size) * max_epochs - global_train_batch + 1))

            ##训练开始：validation
            sess.run(disable_training_op)
            local_valid_loss, local_valid_accuracy = 0, 0  #本epoch中valid_loss和valid_acc值
            for valid_img_minibatch_list, valid_label_minibatch_list in get_minibatches(valid_img_list, valid_label_list, batch_size):
                # minibatch数据读取
                valid_img_batch = get_minibatches_content_img(valid_img_minibatch_list, temp_img_shape[0], temp_img_shape[1])

                #valid操作及指标显示
                sess.run(valid_op, feed_dict={img_batch: valid_img_batch, label_batch: valid_label_minibatch_list})

                local_valid_loss = valid_loss_s.mean_variable
                local_valid_accuracy = valid_accuracy_s.mean_variable
                local_valid_loss_value = local_valid_loss.eval(session=sess)
                local_valid_accuracy_value = local_valid_accuracy.eval(session=sess)

            print('-VALID- epoch: %d | valid_loss: %f valid_acc: %f'
                  % (epoch, local_valid_loss_value, local_valid_accuracy_value))

            #每次valid数据集全部验证完成后，对valid_loss/acc进行存储
            valid_loss_s.add_summary(sess, writer, global_train_batch)
            valid_accuracy_s.add_summary(sess, writer, global_train_batch)

            end_time = time.time()
            print('--EPOCH:%d-- runtime: %.2fs' % (epoch ,end_time - start_time))

            ##模型保存：如果valid_acc大于全局valid_acc，则保存
            if local_valid_accuracy_value > global_valid_accuracy:
                global_valid_accuracy = local_valid_accuracy_value
                saver.save(sess, log_path + '/Model_' + str(epoch) + '.ckpt')
                print('---EPOCH:%d--- model has been saved' % (epoch))

'''按照train方式改动
def test_dataset(model_class, gen, batch_size, ds_size, load_path):
    tf.reset_default_graph()
    runner = GeneratorRunner(gen, batch_size * 10)
    img_batch, label_batch = runner.get_batched_inputs(batch_size)
    model = model_class(False, 'NCHW')
    model._build_model(img_batch)
    loss, accuracy = model._build_losses(label_batch)
    loss_summary = average_summary(loss, 'loss',  \
                                   float(ds_size) / float(batch_size))
    accuracy_summary = average_summary(accuracy, 'accuracy',  \
                                   float(ds_size) / float(batch_size))
    increment_op = tf.group(loss_summary.increment_op, \
                            accuracy_summary.increment_op)
    global_step = tf.get_variable('global_step', dtype=tf.int32, shape=[], \
                                  initializer=tf.constant_initializer(0), \
                                  trainable=False)
    init_op = tf.group(tf.global_variables_initializer(), \
                       tf.local_variables_initializer())
    saver = tf.train.Saver(max_to_keep=10000)
    with tf.Session() as sess:
        sess.run(init_op)
        saver.restore(sess, load_path)
        runner.start_threads(sess, 1)
        for j in range(0, ds_size, batch_size):
            sess.run(increment_op)
        mean_loss, mean_accuracy = sess.run([loss_summary.mean_variable ,\
                                             accuracy_summary.mean_variable])
    print("Accuracy:", mean_accuracy, " | Loss:", mean_loss)

def find_best(model_class, valid_gen, test_gen, valid_batch_size, \
              test_batch_size, valid_ds_size, test_ds_size, load_paths):
    tf.reset_default_graph()
    valid_runner = GeneratorRunner(valid_gen, valid_batch_size * 30)
    img_batch, label_batch = valid_runner.get_batched_inputs(valid_batch_size)
    model = model_class(False, 'NCHW')
    model._build_model(img_batch)
    loss, accuracy = model._build_losses(label_batch)
    loss_summary = average_summary(loss, 'loss',  \
                                          float(valid_ds_size) \
                                          / float(valid_batch_size))
    accuracy_summary = average_summary(accuracy, 'accuracy',  \
                                          float(valid_ds_size) \
                                          / float(valid_batch_size))
    increment_op = tf.group(loss_summary.increment_op, \
                            accuracy_summary.increment_op)
    global_step = tf.get_variable('global_step', dtype=tf.int32, shape=[], \
                                  initializer=tf.constant_initializer(0), \
                                  trainable=False)
    init_op = tf.group(tf.global_variables_initializer(), \
                       tf.local_variables_initializer())
    saver = tf.train.Saver(max_to_keep=10000)
    accuracy_arr = []
    loss_arr = []
    print("validation")
    for load_path in load_paths:
        with tf.Session() as sess:
            sess.run(init_op)
            saver.restore(sess, load_path)
            valid_runner.start_threads(sess, 1)
            _time = time.time()
            for j in range(0, valid_ds_size, valid_batch_size):
                sess.run(increment_op)
            mean_loss, mean_accuracy = sess.run([loss_summary.mean_variable ,\
                                            accuracy_summary.mean_variable])
            accuracy_arr.append(mean_accuracy)
            loss_arr.append(mean_loss)
            print(load_path)
            print("Accuracy:", accuracy_arr[-1], "| Loss:", loss_arr[-1], \
                    "in", time.time() - _time, "seconds.")
    argmax = np.argmax(accuracy_arr)
    print("best savestate:", load_paths[argmax], "with", \
            accuracy_arr[argmax], "accuracy and", loss_arr[argmax], \
            "loss on validation")
    print("test:")
    test_dataset(model_class, test_gen, test_batch_size, test_ds_size, \
                 load_paths[argmax])
    return argmax, accuracy_arr, loss_arr
'''