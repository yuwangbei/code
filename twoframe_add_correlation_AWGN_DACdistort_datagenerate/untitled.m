clc;
clear;
close all;

%============================≤‚ ‘À˘”√======================================
%============================2019.8.20=====================================
load('train_SNR_1.mat','train_SNR_1');
load('train_SNR_2.mat','train_SNR_2');
load('train_SNR_3.mat','train_SNR_3');
load('train_SNR_4.mat','train_SNR_4');

norm_train_SNR_1 = train_SNR_1/norm(train_SNR_1);
norm_train_SNR_2 = train_SNR_2/norm(train_SNR_2);
norm_train_SNR_3 = train_SNR_3/norm(train_SNR_3);
norm_train_SNR_4 = train_SNR_4/norm(train_SNR_4);

save('norm_train_SNR_1','norm_train_SNR_1');
save('norm_train_SNR_2','norm_train_SNR_2');
save('norm_train_SNR_3','norm_train_SNR_3');
save('norm_train_SNR_4','norm_train_SNR_4');
