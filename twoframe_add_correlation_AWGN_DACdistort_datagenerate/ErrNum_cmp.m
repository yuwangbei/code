function err_num = ErrNum_cmp(X_source, X_ELM)




    real_x = real(X_source); imag_x = imag(X_source);
    real_est = real(X_ELM); imag_est = imag(X_ELM);
    
    real_est(find(real_est>=0)) = sqrt(0.5); real_est(find(real_est<0)) = -sqrt(0.5); 
    imag_est(find(imag_est>=0)) = sqrt(0.5); imag_est(find(imag_est<0)) = -sqrt(0.5); 

    temp_real = find(real_est ~= real_x); temp_imag = find(imag_est ~= imag_x); 
    err_num = length(temp_real)+length(temp_imag);
    
    







