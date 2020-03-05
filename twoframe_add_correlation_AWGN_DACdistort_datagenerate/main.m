clc;
clear;
close all;
warning off;

%=========================�۲��֡����֡ͬ��==============================
%�ο����£�[10] Frame Synchronization Based on Multiple Frame Observations
%���������еĲ�������õ���������һ���Ľ��
%===============================2019.12.11==================================

%=========init========
Ns_Chu    = 35;    %---ZC���г���
N_frame   = 100;   %---һ֡�ĳ���
simutime  = 1e5;  %---�������
M_frame   = 4;    %---����4֡����Tao��ʼ��ȡ3֡��������2֡���������
Corr_num  = M_frame - 2;   %---����2֡���������
Switch = 0;       %---0:HPAʧ�棻1:��ʧ�棻2��DACʧ��
Switch_Gen_data = 2;  %---���ݵ��ƿ��� 0�������ƣ�1��BPSK���ƣ�2��QPSK����
alpha = sqrt(2/pi);

%====HPAʧ�����init=======
aa = 1.96;
ba = 0.99;
af = 2.53;
bf = 2.82;
%===I-Q��·��ƽ��ʧ�����init=====
% a1 = 0.78;
% a2 = 0.75;

%=====SNR init=========
SNR_dB      = -6;
Sig_Power   = 1;
Noise_Power = Sig_Power./(10.^(0.1*SNR_dB));   %---��������;
Distort_Power = 0.5*Sig_Power;                 %---��ʧ�����
SDR = alpha^2 * Sig_Power / Distort_Power; 

%======����ZC����======
Chu_seq = Zadoff_Chu(Ns_Chu).';
% Chu_seq = [1, 1, 1, -1, -1, 1, -1].';
Train_seq = [Chu_seq;zeros(N_frame-Ns_Chu,1)];
Corr_matrix = Shift_matrix(N_frame, N_frame, Train_seq);

for SNR_num = 1:length(SNR_dB)
    tic
    correct_num = 0;    %---��ȷ��������
    Rs_frame_conbine = zeros(N_frame,simutime);
    Label_train = zeros(N_frame,simutime);
    
    for jjj= 1:simutime
        TAO = randi([0,N_frame-1],1);   %---ƫ��TAO
        
        Label_train(TAO+1,jjj) = 1;
        %====���ɷ�������=====
        send_data = generateDATA(Ns_Chu, Chu_seq, TAO, N_frame, M_frame, Sig_Power, aa, ba, af, bf, Switch, Switch_Gen_data, alpha);

        %====�ŵ�====
        H_Channel = 1;

        %===����====
        Noise = sqrt(0.5) * (randn((M_frame-1)*N_frame,1) + 1i*randn((M_frame-1)*N_frame,1));

        %===������ʧ����====
%         Nonlinear_Distort = sqrt(0.5) * (randn((M_frame-1)*N_frame,1) + 1i*randn((M_frame-1)*N_frame,1));
        
        %======�����ź�=======
        Receive = send_data * H_Channel + sqrt(Noise_Power(SNR_num))*Noise ;%+ sqrt(Distort_Power)*Nonlinear_Distort;   %---r=Wh + n + d
    
        %===�����źŽ�ȡRs====
        
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
    
    Error_Pro(SNR_num) = 1 - correct_num/simutime;   %---֡�������
    
    toc
end



Double_7_add_DAC_Error_Pro = Error_Pro;
%save('Double_7_add_DAC_Error_Pro.mat','Double_7_add_DAC_Error_Pro');

%============================��ͼ========================================
%============֡ͬ������ɫ��==================
figure(1);
semilogy(SNR_dB,Double_7_add_DAC_Error_Pro,'-b*','LineWidth',2,'MarkerSize',8);
xlabel('SNR');
ylabel('correct probability of frame');
title('֡ͬ��');
axis([-6,12,0,1])






