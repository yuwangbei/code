# coding: utf-8

# 将卷积神经网络应用于帧同步，结构为一个CNN网络+全连接层

import numpy as np
import tensorflow as tf
from Generate_data import *
import pandas as pd
from matplotlib import pyplot as plt

#参数初始化
batch_size = 200  #每个batch的大小
Total_train_num = 100000
Total_test_num = 20000
Input_Node = 100
Output_Node = 100
EPOCHS = 1
train_batch_num = Total_train_num // batch_size  #训练一个epoch有多少个批次
test_batch_num = Total_test_num // batch_size #测试一个epoch有多少个批次
SNR = [1,2,3,4]
LEARNING_RATE_BASE = 0.0001  # 学习速率
LEARNING_RATE_BASE_Vary_times = Total_train_num  # 学习速率多少次变化一次
STAIRCASE = True


# 初始化权值
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)    # 生成一个截断的正态分布（高斯分布）
    return tf.Variable(initial)


# 初始化偏置值
def bias_variable(shape):
    initial = tf.constant(1.0, shape=shape)             # 全部置为0.1
    return tf.Variable(initial)


# 卷积层
def conv2d(x, W):
    # x : input tensor of shape [batch, in_height, in_width, in_channels]，如果是图片，shape就是 [批次大小，高，宽，通道数]
    # W : filter/kernel tensor of shape [filter_height, filter_width, in_channels, out_channels]
    # strides : 步长，必须strides[0] = strides[3] = 1，strides[1]代表x方向上的步长，strides[2]代表y方向上的步长
    # padding : 'SAME'/'VALID'
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='VALID')


# 池化层
def max_pool_2x2(x):
    # ksize : [1, x, y, 1]
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


# 定义两个placeholder，x和y分别用于表示输入的手写数字图像和标签，（具体数据）在执行图的时候喂进来
x = tf.placeholder(tf.float32, [None, Input_Node], name="x_input")     # placeholder(dtype, shape=None, name=None)
y = tf.placeholder(tf.float32, [None, Output_Node], name="y_input")

# 改变x的格式为4D的向量[batch, in_height, in_width, in_channels]，把图片恢复成原来的尺寸28*28
# -1会由实际计算出来的值代替，比如一个batch是100，实际数据量是100*784，那么-1由100*784*1/28/28/1=100代替（通道数为1）
#一维变为二维，取数按行取
#经测试，没有问题，200个10*10的二维数组，按行取，10个为1行，然后再10个第二行，100个取完为1个图形，再来第2个图形
x_image = tf.reshape(x, [-1, 10, 10, 1])    # 对应conv2d函数的参数x，输入的图片数据原来是一维的张量，而conv2d函数要将图片数据恢复成原来的形状

# 初始化第一个卷积层的权值和偏置值
W_conv1 = weight_variable([3, 3, 1, 32])    # 3*3的采样窗口，32个卷积核从1个平面抽取特征（输出32个特征平面）。对应conv2d函数的输入W，函数会将该矩阵reshape成[5*5*1, 32]
b_conv1 = bias_variable([32])               # 每一个卷积核需要一个偏置值

# 把x_image和权值进行卷积，加上偏置值，然后应用于relu激活函数
h_conv1 = tf.math.tanh(conv2d(x_image, W_conv1) + b_conv1)    # 第一个卷积层的结果
h_pool1 = max_pool_2x2(h_conv1)                             # 进行max-pooling，得到池化结果

# 初始化第二个卷积层的权值和偏置值
W_conv2 = weight_variable([2, 2, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.math.tanh(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# 28*28的图片经过第一次卷积之后还是28*28，第一次池化之后变成14*14（14*14*32）
# 第二次卷积之后是14*14，第二次池化之后变成7*7（14*14*64）
# 经过上面的操作之后变成64张7*7的平面
# 图片的维度变高了，第一次卷积池化后由1维变成32维（32个卷积核），第二次变成64维

# 初始化第一个全连接层的权值
W_fc1 = weight_variable([2*2*64, 1024])     # 上一层有7*7*64个神经元，全连接层有1024个神经元
b_fc1 = bias_variable([1024])

# 把池化层2的输出扁平化为1维
h_pool2_flat = tf.reshape(h_pool2, [-1, 2*2*64]) #-1看成x,未知数，根据维度自动计算填充；将h_pool1变为batch_size行，4*4*32列的矩阵
# 求第一个全连接层的输出
h_fc1 = tf.math.tanh(tf.matmul(h_pool2_flat, W_fc1) + b_fc1) #tf.matmul 两个矩阵相乘

# keep_prob用来表示神经元的输出概率
keep_prob = tf.placeholder(tf.float32, name="keep_prob")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)


# 初始化第二个全连接层的权值
W_fc2 = weight_variable([1024, Output_Node])     # 上一层有7*7*64个神经元，全连接层有1024个神经元
b_fc2 = bias_variable([Output_Node])

# 计算输出
predection = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

# predection里面存放的是每个分类的概率，这里返回最大概率所在的位置，即最后的分类结果
result = tf.argmax(predection, 1, name="result") #0为列中最大值的位置；1为行中最大值位置

# 交叉熵代价函数
#tf.nn.softmax_cross_entropy_with_logits的label可以onehot编码
#tf.nn.sparse_softmax_cross_entropy_with_logits自动将label转换成onehot编码形式，label不能用one-hot编码的
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=predection))

# 定义当前迭代轮数的变量
global_step = tf.get_variable('global_step',dtype=tf.int32, initializer=0,trainable=False)  # 不可训练
# 指数衰减学习速率
# learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE,
#                                            global_step,
#                                            LEARNING_RATE_BASE_Vary_times,
#                                            0.96,
#                                            staircase=STAIRCASE)

# 使用AdamOptimizer进行优化，学习率是1e-4
train_step = tf.train.AdamOptimizer(LEARNING_RATE_BASE).minimize(loss,global_step)
# 结果存放在一个布尔列表中
correct_predection = tf.equal(tf.argmax(predection, 1), tf.argmax(y, 1))    # argmax返回一维张量中最大值所在的位置
# 求准确率
accuracy = tf.reduce_mean(tf.cast(correct_predection, tf.float32))

saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    Total_error_probability = list(range(len(SNR)))
    for snr in SNR:
        train_data = read_train_data(snr).T
        train_label = read_train_label(snr).T
        test_data = read_test_data(snr).T
        test_label = read_test_label(snr).T

        accuracy_mean_list = []
 #========================训练部分=========================================================
        for epoch in range(EPOCHS):
            for batch in range(train_batch_num):
                batch_xs, batch_ys = data_choose(train_data, train_label, batch_size)
                sess.run(train_step, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 0.5})#
                train_loss = sess.run(loss, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1.0})#
                train_accuracy = sess.run(accuracy, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1.0})#

                    #=======================测试所用==========================
                    # h_pool2_flat = sess.run(h_pool2_flat, feed_dict={x: batch_xs, y: batch_ys})
                    # print(h_pool2_flat)
                    # print(h_pool2_flat.shape)
                # h_pool2 = sess.run(h_pool2, feed_dict={x: batch_xs, y: batch_ys})
                # print(h_pool2.shape)
                # predection = sess.run(predection, feed_dict={x: batch_xs, y: batch_ys})
                # acs1 = sess.run(tf.argmax(y, 1), feed_dict={x: batch_xs, y: batch_ys})
                # acs2 = sess.run(tf.argmax(predection, 1), feed_dict={x: batch_xs, y: batch_ys})

                    # x_image = sess.run(x_image, feed_dict={x: batch_xs, y: batch_ys})
                    # x = sess.run(x, feed_dict={x: batch_xs, y: batch_ys})
                    #h_conv1 = sess.run(h_conv1, feed_dict={x: batch_xs, y: batch_ys})
                    #b_conv1_test = sess.run(b_conv1, feed_dict={x: batch_xs, y: batch_ys})
                    #W_conv1_test = sess.run(W_conv1, feed_dict={x: batch_xs, y: batch_ys})
                    # print(h_conv1.shape)
                    # print(b_conv1.shape)
                    #print(W_conv1.shape)


                if batch % 100 == 0:
                    print('SNR:', snr)
                    print('Epoch:', epoch)
                    print('训练:%d' % batch)
                    print('训练校验集损失:%.12f' % train_loss)
                    print('训练准确率：', train_accuracy)


            #========================测试部分===========================================================
            for batch in range(test_batch_num):
                    batch_xs, batch_ys = data_choose(test_data, test_label, batch_size)
                    sess.run(train_step, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 0.5})
                    test_loss = sess.run(loss, feed_dict = {x: batch_xs, y: batch_ys, keep_prob: 1.0})
                    test_accuracy = sess.run(accuracy, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1.0})

                    # if batch % 50 == 0:
                    #     print('测试:%d' % batch)
                    #     print('测试集损失:%.12f' % test_loss)
                    #     print('测试准确率为%d' % test_accuracy)

                    if epoch == EPOCHS - 1:
                       if batch >= 50:
                           accuracy_mean_list.append(1-test_accuracy)

        Total_error_probability[snr-1] = sum(accuracy_mean_list)/len(accuracy_mean_list)
        saver.save(sess, 'model/cnn_model', global_step=snr)

    print(Total_error_probability)

SNR = np.array(SNR)
SNR = (SNR-2)*6
pd.DataFrame(data=Total_error_probability, index=SNR).plot()
plt.title('CNN FS')
plt.xlabel('SNR')
plt.ylabel('Error probability of FS')
plt.grid()
plt.savefig('F:\Python_project\CNN_Twoframe_frame\CNN_FS.pdf')
plt.show()




