import tensorflow as tf
import numpy as np
from scipy import misc, io
import time
from glob import glob
from generator import *

# *包含loss与acc变量及操作的average_summary类
class average_summary(object):
    def __init__(self, variable, name, num_iterations):
        # sum_variable：内在累加器，用于累加每次的loss/acc
        self.sum_variable = tf.get_variable(name, shape=[],
                                            initializer=tf.constant_initializer(0.),
                                            dtype='float32',
                                            trainable=False,
                                            collections=[tf.GraphKeys.LOCAL_VARIABLES])
        # 每个batch调用一次increment_op，累加每次的loss/acc
        with tf.control_dependencies([variable]):
            self.increment_op = tf.assign_add(self.sum_variable, variable)
        # 当increment_op操作调用了num_iterations次之后，可进行下列操作
        self.mean_variable = self.sum_variable / float(num_iterations)  # 求平均的loss和acc
        self.summary = tf.summary.scalar(name, self.mean_variable)  # 将loss和acc存入tf全局图
        with tf.control_dependencies([self.summary]):
            self.reset_variable_op = tf.assign(self.sum_variable, 0)  # 当summary完成后，可进行reset
    # 外部调用，将loss/acc存入tf全局图
    def add_summary(self, sess, writer, step):
        s, _ = sess.run([self.summary, self.reset_variable_op])
        writer.add_summary(s, step)

# *用于挂载Net的结构，包含__build_model和__build_loss的操作
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
            oh = tf.one_hot(self.labels, 2)  # 这里定义了2分类的输出
            # *除softmax cross entropy之外，还可更换其他函数
            self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                                       labels=oh, logits=self.outputs))
        with tf.variable_scope('accuracy'):
            am = tf.argmax(self.outputs, 1)
            equal = tf.equal(am, self.labels)
            self.accuracy = tf.reduce_mean(tf.cast(equal, tf.float32))
        return self.loss, self.accuracy

# *训练主函数
def train(model_class, use_batch_norm, use_shuf_pair,
          train_cover_dir, train_stego_dir,
          valid_cover_dir, valid_stego_dir,
          batch_size, train_ds_size, valid_ds_size,
          log_interval, max_epochs, lr,
          log_path, load_path=None):
    # *清除默认图的堆栈，设置全局图为默认图
    tf.reset_default_graph()

    # *is_training用于判断训练处于train或者valid状态
    is_training = tf.get_variable('is_training', dtype=tf.bool,
                                  initializer=True, trainable=False)

    # *定义train_op操作和valid_op操作，将is_training和batch_size设置为对应的状态
    disable_training_op = tf.assign(is_training, False)
    enable_training_op = tf.assign(is_training, True)

    # *模型初始化
    # 设置占位符
    temp_cover_list = glob(train_cover_dir + '/*')
    temp_img = misc.imread(temp_cover_list[0])
    temp_img_shape = temp_img.shape
    img_batch = tf.placeholder(tf.float32,
                               [batch_size, temp_img_shape[0], temp_img_shape[1], 1],
                               name='input_image_batch')
    label_batch = tf.placeholder(tf.int32, [batch_size, ], name="input_label_batch")
    # 使用占位符初始化模型
    model = model_class(is_training=is_training, data_format='NCHW',
                        with_bn=use_batch_norm, tlu_threshold=3)
    model._build_model(img_batch)
    loss, accuracy = model._build_losses(label_batch)

    # *设置需要最小化的loss函数
    regularization_losses = tf.get_collection(
                          tf.GraphKeys.REGULARIZATION_LOSSES)
    regularized_loss = tf.add_n([loss] + regularization_losses)
    # 定义train中使用的基于loss/acc的类（运行次数：log_interval）
    train_loss_s = average_summary(loss, 'train_loss', log_interval)
    train_accuracy_s = average_summary(accuracy, 'train_accuracy', log_interval)
    # 定义valid中使用的基于loss/acc的类（运行次数：valid_ds_size / valid_batch_size）
    valid_loss_s = average_summary(loss, 'valid_loss',
                                   float(valid_ds_size) / float(batch_size))
    valid_accuracy_s = average_summary(accuracy, 'valid_accuracy',
                                       float(valid_ds_size) / float(batch_size))

    # *全局变量global_step，从0开始进行计数
    global_step = tf.Variable(0, trainable=False)
    # *定义核心optimizer
    # 定义learning_rate的decay操作
    init_learning_rate = lr
    decay_steps, decay_rate = 2000, 0.95
    learning_rate = learning_rate_decay(init_learning_rate=init_learning_rate,
                                        decay_method="exponential",
                                        global_step=global_step,
                                        decay_steps=decay_steps,
                                        decay_rate=decay_rate)
    optimizer = tf.train.AdadeltaOptimizer(learning_rate)

    # *定义train及valid过程中需要用到的操作
    # 核心操作：最小化loss
    minimize_op = optimizer.minimize(loss=regularized_loss, global_step=global_step)
    # 训练操作（每个iteration都要用）：最小化loss；train_loss累加；train_acc累加
    train_op = tf.group(minimize_op, train_loss_s.increment_op,
                        train_accuracy_s.increment_op)
    # 验证操作（一个epoch结束后，每个valid中的iteration都要用）：valid_loss累加；valid_acc累加
    valid_op = tf.group(valid_loss_s.increment_op,
                        valid_accuracy_s.increment_op)
    # 初始化操作：初始化所有的全局变量和局部变量
    init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())

    # *定义模型保存变量，最大存储max_to_keep个模型
    saver = tf.train.Saver(max_to_keep=max_epochs)
    global_valid_accuracy = 0  # 全局valid_acc最大值

    # *会话开始
    with tf.Session() as sess:
        # 初始化所有的全局变量和局部变量
        sess.run(init_op)
        # 重载模型
        if load_path is not None:
            loader = tf.train.Saver(reshape=True)
            loader.restore(sess, load_path)
        # 定义模型及参数保存位置
        writer = tf.summary.FileWriter(log_path + '/LogFile/', sess.graph)

        # 初始化train/valid的loss和acc变量
        sess.run([valid_loss_s.reset_variable_op,
                  valid_accuracy_s.reset_variable_op,
                  train_loss_s.reset_variable_op,
                  train_accuracy_s.reset_variable_op])

        # *训练开始：train/valid
        print('Start training...')
        global_train_batch = 0  # 全局batch计数
        for epoch in range(max_epochs):
            start_time = time.time()
            # 加载test路径下的img及label列表
            train_img_list, train_label_list = get_files(train_cover_dir,
                                                         train_stego_dir,
                                                         use_shuf_pair=use_shuf_pair)
            # 加载valid路径下的img及label列表
            valid_img_list, valid_label_list = get_files(valid_cover_dir,
                                                         valid_stego_dir,
                                                         use_shuf_pair=use_shuf_pair)

            # *训练开始：train
            sess.run(enable_training_op)  # 转换为train训练状态
            local_train_batch = 0  # 局部batch计数
            for train_img_minibatch_list, train_label_minibatch_list in \
                    get_minibatches(train_img_list, train_label_list, batch_size):
                # minibatch数据读取
                train_img_batch = get_minibatches_content_img(train_img_minibatch_list,
                                                              temp_img_shape[0],
                                                              temp_img_shape[1])

                # train操作及指标显示
                sess.run(train_op, feed_dict={img_batch: train_img_batch,
                                              label_batch: train_label_minibatch_list})

                global_train_batch += 1
                local_train_batch += 1

                # 每log_interval个batch后，对train_loss/acc进行存储
                # 这是由于train_loss/acc的average_summary以log_interval为基准定义
                if global_train_batch % log_interval == 0:
                    # 注意：loginterval决定了每20输出一次，而不是每个batch存储loss/acc一次
                    # train_loss/acc显示
                    local_train_loss = train_loss_s.mean_variable
                    local_train_accuracy = train_accuracy_s.mean_variable
                    local_train_loss_value = local_train_loss.eval(session=sess)
                    local_train_accuracy_value = local_train_accuracy.eval(session=sess)
                    print('-TRAIN- epoch: %d batch: %d | train_loss: %f train_acc: %f'
                          % (epoch, local_train_batch, local_train_loss_value, local_train_accuracy_value))
                    # train_loss/acc存储
                    train_loss_s.add_summary(sess, writer, global_train_batch)
                    train_accuracy_s.add_summary(sess, writer, global_train_batch)

            # *训练开始：validation
            sess.run(disable_training_op)
            local_valid_loss, local_valid_accuracy = 0, 0  # 本epoch中valid_loss和valid_acc值
            for valid_img_minibatch_list, valid_label_minibatch_list in \
                    get_minibatches(valid_img_list, valid_label_list, batch_size):
                # minibatch数据读取
                valid_img_batch = get_minibatches_content_img(valid_img_minibatch_list,
                                                              temp_img_shape[0],
                                                              temp_img_shape[1])

                # valid操作及指标显示
                sess.run(valid_op, feed_dict={img_batch: valid_img_batch,
                                              label_batch: valid_label_minibatch_list})

            # 每个epoch中所有batch运行完后，对valid_loss/acc进行显示和存储
            # 这是由于valid_loss/acc的average_summary以(valid_ds_size/batch_size)为基准定义
            # valid_loss/acc显示
            local_valid_loss = valid_loss_s.mean_variable
            local_valid_accuracy = valid_accuracy_s.mean_variable
            local_valid_loss_value = local_valid_loss.eval(session=sess)
            local_valid_accuracy_value = local_valid_accuracy.eval(session=sess)
            print('-VALID- epoch: %d | valid_loss: %f valid_acc: %f'
                  % (epoch, local_valid_loss_value, local_valid_accuracy_value))
            # valid_loss/acc存储
            valid_loss_s.add_summary(sess, writer, global_train_batch)
            valid_accuracy_s.add_summary(sess, writer, global_train_batch)

            # *模型保存：如果valid_acc大于全局valid_acc，则保存
            if local_valid_accuracy_value > global_valid_accuracy or (max_epochs - epoch) < 5:
                global_valid_accuracy = local_valid_accuracy_value
                saver.save(sess, log_path + '/Model_' + str(epoch) + '.ckpt')
                print('---EPOCH:%d--- model has been saved' % epoch)

            # *本epoch中train及valid过程均完毕，记录时间
            end_time = time.time()
            print('--EPOCH:%d-- runtime: %.2fs ' % (epoch, end_time - start_time),
                  ' learning rate: ', sess.run(learning_rate), '\n')

# *测试主函数，查找最佳模型
def test_dataset_findbest(model_class, use_shuf_pair,
                          test_cover_dir, test_stego_dir, max_epochs,
                          batch_size, ds_size, log_path):
    tf.reset_default_graph()

    # *模型初始化
    # 设置占位符
    temp_cover_list = glob(test_cover_dir + '/*')
    temp_img = misc.imread(temp_cover_list[0])
    temp_img_shape = temp_img.shape
    img_batch = tf.placeholder(tf.float32,
                               [batch_size, temp_img_shape[0], temp_img_shape[1], 1],
                               name='input_image_batch')
    label_batch = tf.placeholder(tf.int32, [batch_size, ], name="input_label_batch")
    # 使用占位符初始化模型
    model = model_class(is_training=False, data_format='NCHW', with_bn=True, tlu_threshold=3)
    model._build_model(img_batch)
    loss, accuracy = model._build_losses(label_batch)

    # *设置需要计算的loss函数，test_loss/acc与valid_loss/acc的功用类似
    # 定义valid中使用的基于loss/acc的类（运行次数：valid_ds_size / valid_batch_size）
    test_loss_s = average_summary(loss, 'test_loss',
                                  float(ds_size) / float(batch_size))
    test_accuracy_s = average_summary(accuracy, 'test_accuracy',
                                      float(ds_size) / float(batch_size))
    # 验证操作（一个epoch结束后，每个valid中的iteration都要用）：valid_loss累加；valid_acc累加
    test_op = tf.group(test_loss_s.increment_op,
                       test_accuracy_s.increment_op)

    # 初始化操作：初始化所有的全局变量和局部变量
    init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())

    # *定义模型保存变量，最大存储max_to_keep个模型
    saver = tf.train.Saver()

    # *记录每次test后得到的loss和acc
    test_loss_arr = []
    test_accuracy_arr = []

    # *对load_data_path_s列表中的所有模型进行test操作
    print('Start testing...')
    # 在log路径下搜寻所有可加载文件
    load_model_path_s = glob(log_path + '/*.data*')
    for load_model_path in load_model_path_s:
        start_time = time.time()
        # *会话开始
        with tf.Session() as sess:
            # 初始化所有的全局变量和局部变量
            sess.run(init_op)
            # 重载模型，去掉结尾的.data-000...
            trunc_str = '.data-'
            load_model_path_trunc = load_model_path[0:load_model_path.find(trunc_str)]
            saver.restore(sess, load_model_path_trunc)
            # 初始化test的loss和acc变量
            sess.run([test_loss_s.reset_variable_op,
                      test_accuracy_s.reset_variable_op])
            # 加载test路径下的img及label列表
            test_img_list, test_label_list = get_files(test_cover_dir,
                                                       test_stego_dir,
                                                       use_shuf_pair=use_shuf_pair)
            # *对当前load_data_path的模型进行test操作
            for test_img_minibatch_list, test_label_minibatch_list in \
                    get_minibatches(test_img_list, test_label_list, batch_size):
                # minibatch数据读取
                test_img_batch = get_minibatches_content_img(test_img_minibatch_list,
                                                             temp_img_shape[0],
                                                             temp_img_shape[1])
                # 对每次minibatch中test后得到的loss和acc进行累加
                sess.run(test_op, feed_dict={img_batch: test_img_batch,
                                             label_batch: test_label_minibatch_list})
            # *记录当前load_data_path模型test操作后得到的loss和acc
            test_mean_loss, test_mean_accuracy = sess.run([test_loss_s.mean_variable,
                                                           test_accuracy_s.mean_variable])
            test_loss_arr.append(test_mean_loss)
            test_accuracy_arr.append(test_mean_accuracy)
            end_time = time.time()
            print(load_model_path.split("/")[-1])
            print('-TEST- test_loss: %f test_acc: %f | runtime: %.2fs \n'
                  % (test_loss_arr[-1], test_accuracy_arr[-1], end_time - start_time))

    # *寻找最佳test_acc对应的模型索引
    load_best_model_idx = np.argmax(test_accuracy_arr)
    print('-BEST TEST- best_path: ', load_model_path_s[load_best_model_idx])
    print('-BEST TEST- best_loss: %f best_acc: %f \n'
          % (test_loss_arr[load_best_model_idx], test_accuracy_arr[load_best_model_idx]))

    return load_model_path_s[load_best_model_idx]


# *学习率下降函数，包含各类学习率下降方法
def learning_rate_decay(init_learning_rate, global_step, decay_steps, decay_rate,
                        decay_method="exponential", staircase=False,
                        end_learning_rate=0.0001, power=1.0, cycle=False,):
    """
    传入初始learning_rate，根据参数及选项运用不同decay策略更新learning_rate
        learning_rate : 初始的learning rate
        global_step : 全局的step，与 decay_step 和 decay_rate一起决定了 learning rate的变化
        staircase : 如果为 True global_step/decay_step 向下取整
        end_learning_rate，power，cycle：只在polynomial_decay方法中使用
    """
    if decay_method == 'constant':
        decayed_learning_rate = init_learning_rate
    elif decay_method == 'exponential':
        decayed_learning_rate = tf.train.exponential_decay(init_learning_rate, global_step,
                                                           decay_steps, decay_rate, staircase)
    elif decay_method == 'inverse_time':
        decayed_learning_rate = tf.train.inverse_time_decay(init_learning_rate, global_step,
                                                            decay_steps, decay_rate, staircase)
    elif decay_method == 'natural_exp':
        decayed_learning_rate = tf.train.natural_exp_decay(init_learning_rate, global_step,
                                                           decay_steps, decay_rate, staircase)
    elif decay_method == 'polynomial':
        decayed_learning_rate = tf.train.polynomial_decay(init_learning_rate, global_step,
                                                          decay_steps, decay_rate,
                                                          end_learning_rate, power, cycle)
    else:
        decayed_learning_rate = init_learning_rate

    return decayed_learning_rate

