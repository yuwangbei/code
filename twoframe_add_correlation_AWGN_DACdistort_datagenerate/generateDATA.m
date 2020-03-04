function shiftW = generateDATA(Ns, Chu_seq, TAO, N, M_frame, Sig_Power, aa, ba, af, bf, Switch, Switch_Gen_data, alpha)

%========================   ������������X����ƫ��   ========================
%===����NsΪѵ�����еĴ�С
%===����Chu_seqΪѵ������
%===����TAOΪƫ��
%===����NΪһ֡���ݳ���
%===����LΪ�ྶ��Ŀ
%========================  2019.8.20   ====================================

%========����========
% clc;
% clear;
% close all;
%===================

%---���� M_frame ֡���ݳ�---
%---0Ϊ������---
if Switch_Gen_data == 0
    M_frame_data = sqrt(0.5) * (randn(M_frame*N,1) + 1i * randn(M_frame*N,1));

%---1ΪBPSK����---    
elseif Switch_Gen_data == 1
    M_frame_data = randn(M_frame*N,1);
    M_frame_data(find(M_frame_data>0)) = 1;
    M_frame_data(find(M_frame_data<=0)) = -1;
    
%---2ΪQPSK����---
elseif Switch_Gen_data == 2
    M_frame_data = sqrt(0.5) * ( (2*randi([0, 1], M_frame*N,1)-1) + 1i * (2*randi([0, 1], M_frame*N,1)-1) );
end

%---����ZC����---
for ii = 0:M_frame-1
    M_frame_data(ii*N+1:ii*N+Ns) = Chu_seq;
end

if Switch == 0
    TT_Receive_matrix = M_frame_data;
    AMP_TT = abs(TT_Receive_matrix);
    PHA_TT = angle(TT_Receive_matrix);

    TT_dataAMP = aa*AMP_TT./(1+ba*AMP_TT.^2);
    TT_dataPHA = af*AMP_TT.^2./(1+bf*AMP_TT.^2);

    data_finally = TT_dataAMP .* exp(1i*TT_dataPHA);
    
%---switch=1,��ʧ��    
elseif Switch == 1
    data_finally = M_frame_data;    
elseif Switch == 2
%     data_finally =  a1 * real(M_frame_data) +  1i * a2 * imag(M_frame_data); 
    data_finally = alpha * M_frame_data;
end
send_data = sqrt(Sig_Power)*data_finally;    

%---����ƫ��TAO��ȡ����W---
shiftW = send_data(TAO+1:TAO+(M_frame-1)*N);







