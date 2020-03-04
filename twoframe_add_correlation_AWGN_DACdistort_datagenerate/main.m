clc;
clear;
close all;
warning off;

%=========================观察多帧进行帧同步==============================
%参考文章：[10] Frame Synchronization Based on Multiple Frame Observations
%按照文章中的参数仿真得到和文章中一样的结果
%===============================2019.12.11==================================

%=========init========
Ns_Chu    = 35;    %---ZC序列长度
N_frame   = 100;   %---一帧的长度
simutime  = 1e5;  %---仿真次数
M_frame   = 4;    %---生成4帧，从Tao开始截取3帧，用其中2帧作相关运算
Corr_num  = M_frame - 2;   %---其中2帧作相关运算
Switch = 0;       %---0:HPA失真；1:无失真；2：DAC失真
Switch_Gen_data = 2;  %---数据调制开关 0：不调制；1：BPSK调制；2：QPSK调制
alpha = sqrt(2/pi);

%====HPA失真参数init=======
aa = 1.96;
ba = 0.99;
af = 2.53;
bf = 2.82;
%===I-Q两路不平衡失真参数init=====
% a1 = 0.78;
% a2 = 0.75;

%=====SNR init=========
SNR_dB      = -6;
Sig_Power   = 1;
Noise_Power = Sig_Power./(10.^(0.1*SNR_dB));   %---噪声功率;
Distort_Power = 0.5*Sig_Power;                 %---非失真项功率
SDR = alpha^2 * Sig_Power / Distort_Power; 

%======生成ZC序列======
Chu_seq = Zadoff_Chu(Ns_Chu).';
% Chu_seq = [1, 1, 1, -1, -1, 1, -1].';
Train_seq = [Chu_seq;zeros(N_frame-Ns_Chu,1)];
Corr_matrix = Shift_matrix(N_frame, N_frame, Train_seq);

for SNR_num = 1:length(SNR_dB)
    tic
    correct_num = 0;    %---正确次数计算
    Rs_frame_conbine = zeros(N_frame,simutime);
    Label_train = zeros(N_frame,simutime);
    
    for jjj= 1:simutime
        TAO = randi([0,N_frame-1],1);   %---偏移TAO
        
        Label_train(TAO+1,jjj) = 1;
        %====生成发送数据=====
        send_data = generateDATA(Ns_Chu, Chu_seq, TAO, N_frame, M_frame, Sig_Power, aa, ba, af, bf, Switch, Switch_Gen_data, alpha);

        %====信道====
        H_Channel = 1;

        %===噪声====
        Noise = sqrt(0.5) * (randn((M_frame-1)*N_frame,1) + 1i*randn((M_frame-1)*N_frame,1));

        %===非线性失真项====
%         Nonlinear_Distort = sqrt(0.5) * (randn((M_frame-1)*N_frame,1) + 1i*randn((M_frame-1)*N_frame,1));
        
        %======接收信号=======
        Receive = send_data * H_Channel + sqrt(Noise_Power(SNR_num))*Noise ;%+ sqrt(Distort_Power)*Nonlinear_Distort;   %---r=Wh + n + d
    
        %===接收信号截取Rs====
        
        Rs_frame1 = Receive(1:N_frame);
        Rs_frame2 = Receive(N_frame+1:N_frame+N_frame);
        Rs_frame3 = Rs_frame1 +  Rs_frame2;
        
        

        MM = abs(Corr_matrix * Rs_frame3).^2;
        Rs_frame_conbine(:,jjj) = MM;
        [~,pos] = max(MM);
        
        

        if pos-1 == TAO
            correct_num = correct_num + 1;
        end
  
    end
    
    HPA_Train_SNR_2 = Rs_frame_conbine;
    HPA_Train_Label_SNR_2 = Label_train;
    
    save('HPA_Train_SNR_2.mat','HPA_Train_SNR_2');
    save('HPA_Train_Label_SNR_2.mat','HPA_Train_Label_SNR_2');
    
    Error_Pro(SNR_num) = 1 - correct_num/simutime;   %---帧错误概率
    
    toc
end



Double_7_add_DAC_Error_Pro = Error_Pro;
%save('Double_7_add_DAC_Error_Pro.mat','Double_7_add_DAC_Error_Pro');

%============================画图========================================
%============帧同步（蓝色）==================
figure(1);
semilogy(SNR_dB,Double_7_add_DAC_Error_Pro,'-b*','LineWidth',2,'MarkerSize',8);
xlabel('SNR');
ylabel('correct probability of frame');
title('帧同步');
axis([-6,12,0,1])






