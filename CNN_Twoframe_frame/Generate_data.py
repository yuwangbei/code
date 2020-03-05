# coding: utf-8

import scipy.io as scio
import numpy as np

def read_train_data(snr):
    Data = scio.loadmat('F://Python_project\CNN_Twoframe_frame\Train_data/train_SNR_%d.mat' % snr)
    x_train = Data['train_SNR_%d' % snr]
    return x_train

def read_norm_train_data(snr):
    Data = scio.loadmat('F:\Python_project\CNN_Twoframe_frame/Norm_Train_data/norm_train_SNR_%d.mat' % snr)
    x_norm_train = Data['norm_train_SNR_%d' % snr]
    return x_norm_train

def read_test_data(snr):
    Data = scio.loadmat('F:\Python_project\CNN_Twoframe_frame\Test_data/test_SNR_%d.mat' % snr)
    x_test = Data['test_SNR_%d' % snr]
    return x_test

def read_train_label(snr):
    label = scio.loadmat('F:\Python_project\CNN_Twoframe_frame\Train_data/Label_train_SNR_%d.mat' % snr)
    y_train_label = label['Label_train_SNR_%d' % snr]
    return y_train_label

def read_test_label(snr):
    label = scio.loadmat('F:\Python_project\CNN_Twoframe_frame\Test_data\Label_test_SNR_%d.mat' % snr)
    y_test_label = label['Label_test_SNR_%d' % snr]
    return y_test_label

def data_choose(data, label, length):
    rand_index = np.random.choice(data.shape[0],size=length)
    output_data = data[rand_index, :]
    output_label = label[rand_index, :]
    return output_data, output_label

