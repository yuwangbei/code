

function Out = Shift_matrix(M, N_frame, Train_seq)
%=====test=====
% clear;
% clc;
% close all;
%=====2019.12.17=====

    Out = zeros(M, N_frame);
    Out(:,1) = Train_seq;
    for iii = 2:N_frame
        Out(:,iii) = circshift( Out(:,1), -(iii-1) );  
    end
    Out = Out';




