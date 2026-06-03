#include "MaxPool2D.hpp"

maxpool2D::maxpool2D(int kernel_size, int stride):kernel_size(kernel_size), stride(stride){};

std:: string maxpool2D::name()const{
    return "maxpool2D";
}

std:: vector<int> maxpool2D::output_shape(const std::vector<int>&input_shape)const{
    if(input_shape.size()!=3){
        throw std::runtime_error("MaxPool2D expects input shape (C,H,W)");  
    }
    std:: vector<int> out;
    int C= input_shape[0];
    int H= input_shape[1];
    int W= input_shape[2];
    int K= kernel_size;
    int S= stride;
    int H_out= (H-K)/S+1;
    int W_out= (W-K)/S +1;
    if(H_out<=0 || W_out <=0){
        throw std::runtime_error("MaxPool2D expects a correct combo of K and S");  
    }
    out.push_back(C);
    out.push_back(H_out);
    out.push_back(W_out);
    return out;
}

Tensor maxpool2D::forward(const Tensor& input){
    throw std:: runtime_error("maxpool2D fwd to be implemented");
}

long long maxpool2D:: parameter_count() const{
    return 0;
}