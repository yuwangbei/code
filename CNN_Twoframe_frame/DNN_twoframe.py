# coding: utf-8


import tensorflow as tf
import keras.backend as K
import keras
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import os
import time
from Generate_data import *

INPUT_NODE = 100  # 输入节点
OUTPUT_NODE = 100  # 输出节点
BATCH_SIZE = 200  # 一批多少数据
EPOCHS = 20  # 过多少次数据
LEARNING_RATE_BASE = 0.0001  # 基本学习率
STAIRCASE = True  # 学习率衰减阶梯状衰减
DROP_OUT = 0.3  # dropout率
Total_train_num = 100000
Total_test_num = 20000
SNR = [0, 1, 2, 3, 4]

config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # 不全部占满显存, 按需分配


# 建模
x = tf.placeholder(tf.float32, shape=(None, INPUT_NODE))
y_ = tf.placeholder(tf.float32, shape=(None, OUTPUT_NODE))

# x = keras.layers.Input(shape=(INPUT_NODE,))
layer1 = keras.layers.Dense(100, activation='relu')(x)

layer1 = keras.layers.Dropout(DROP_OUT)(layer1)

layer2 = keras.layers.Dense(200, activation='relu')(layer1)

layer2 = keras.layers.Dropout(DROP_OUT)(layer2)

layer3 = keras.layers.Dense(100, activation='relu')(layer2)

layer3 = keras.layers.Dropout(DROP_OUT)(layer3)

y = keras.layers.Dense(OUTPUT_NODE, activation='softmax')(layer3)

# model = keras.models.Sequential()
# model.add(keras.layers.normalization.BatchNormalization(input_shape=(INPUT_NODE,)))
# model.add(keras.layers.Dense(128, activation='relu'))
# model.add(keras.layers.Dropout(DROP_OUT))
# model.add(keras.layers.Dense(256, activation='relu'))
# model.add(keras.layers.Dropout(DROP_OUT))
# model.add(keras.layers.Dense(128, activation='relu'))
# model.add(keras.layers.Dropout(DROP_OUT))
# model.add(keras.layers.Dense(OUTPUT_NODE, activation='softmax'))


# def error_mean(y_ture, y_pred):
#     return tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y_ture, 1), tf.argmax(y_pred, 1)),
#                                   dtype=tf.int32))

# def error_num(y_ture, y_pred):
#     return keras.backend.sum(keras.backend.square(y_pred - y_ture))
#
# model = keras.models.Model(inputs=x, outputs=y)
#
# model.compile(loss='categorical_crossentropy',
#               optimizer=keras.optimizers.Adam(LEARNING_RATE_BASE, beta_1=0.9, beta_2=0.999, epsilon=1e-08),
#               metrics=['accuracy', error_mean])


loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y,
                                                                     labels=tf.argmax(y_, 1)))

# loss = - tf.reduce_mean(y_ * tf.log(tf.clip_by_value(y, 1e-8, 1e2)))

accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1)), dtype=tf.float32))

# 优化器
# 定义当前迭代轮数的变量
global_step = tf.get_variable('global_step',  # 存储当前迭代的轮数
                              dtype=tf.int32,  # 整数
                              initializer=0,  # 初始化值
                              trainable=False)  # 不可训练

# 指数衰减学习速率，随着训练的进行逐步降低学习率（训练初始用大学习率，后期衰减更加稳定）
learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE,
                                           global_step,
                                           Total_train_num / BATCH_SIZE,
                                           0.99,
                                           staircase=STAIRCASE)

# 定义优化函数
train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step)

# history_list = []
# time_use = []
# for snr in SNR:
#     print('SNR:%d'%snr)
#     start_time = time.time()
#     X_train, X_test, Y_train, Y_test = read_data(snr)
#     history = model.fit(
#         X_train,
#         Y_train,
#         batch_size=BATCH_SIZE,
#         epochs=EPOCHS,
#         verbose=1,
#         validation_data=(X_test, Y_test),
#         )
#     history_list.append(history)
#     end_time = time.time()
#     time_use.append(end_time-start_time)
#     del(X_train, X_test, Y_train, Y_test)
#
# print(time_use)
#



with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    #for snr in SNR:
    train_data = read_train_data(1).T
    train_label = read_train_label(1).T

    # 保存开始时间
    #time_start = time.time()
    sess.run(tf.global_variables_initializer())  # 初始化
    for epoch in range(EPOCHS):
        for batch in range(Total_train_num // BATCH_SIZE):
            batch_xs, batch_ys = data_choose(train_data, train_label, BATCH_SIZE)
            sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

            if batch % 50 == 0:
                print(batch)
                print('校验集损失:%.12f' % sess.run(loss, feed_dict={x: batch_xs, y_: batch_ys}) )
                print('训练准确率：', sess.run(accuracy, feed_dict={x: batch_xs, y_: batch_ys}))


# pd.DataFrame(data=train_loss_list, index=SNR).T.plot()
# plt.title('Train loss')
# plt.grid()
# plt.xlabel('steps')
# plt.ylabel('loss')
# plt.savefig(r'./train_loss/train_loss' + '%s' % (time.strftime('%d_%H_%M')) + '.jpg')
# plt.show()
#
# pd.DataFrame(data=test_loss_list, index=SNR).T.plot()
# plt.title('Test loss')
# plt.grid()
# plt.xlabel('steps')
# plt.ylabel('loss')
# plt.savefig('./test_loss/test_loss' + '%s' % (time.strftime('%d_%H_%M')) + '.jpg')
# plt.show()
#
# pd.Series(data=time_list, index=[0, 10, 20, 30, 40]).plot()
# plt.title('time using')
# plt.grid()
# plt.xlabel('SNR')
# plt.ylabel('time/s')
# plt.savefig('./time_use/time_use' + '%s' % (time.strftime('%d_%H_%M')) + '.jpg')
# plt.show()
