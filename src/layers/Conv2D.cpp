
// Input: 3 channels (RGB image)
//        ↓
//     [Filter of shape (3, k, k)]
//        ↓
//     Apply convolution across all 3 channels
//        ↓
//     Sum the results across channels
//        ↓
//     1 output feature map


#include "Conv2D.hpp"

std:: string conv2D::name()const{
    return "conv2D";
}
conv2D ::conv2D(int in_channel,int out_channel,int kernel_size,int stride,int padding):
in_channel(in_channel),out_channel(out_channel),kernel_size(kernel_size),stride(stride),padding(padding)
{}
std::vector<int>  conv2D::output_shape(const std:: vector<int> &input_shape)const{
    if(input_shape.size() != 3){
    throw std::runtime_error("Conv2D expects input shape (C,H,W)");}
    std:: vector<int>out;
    int C = input_shape[0];
    if(C != in_channel){
    throw std::runtime_error("Input channel mismatch");}
    int H = input_shape[1];
    int W = input_shape[2];
    int P= padding;
    int S= stride;
    int K= kernel_size;
    int H_out= (H+ 2*P -K)/S + 1; 
    int W_out= (W+ 2*P -K)/S + 1; 
    if(H_out<=0 || W_out<=0){
        throw std:: runtime_error("invalid Conv2d output shape");
    }
    out.push_back(out_channel);
    out.push_back(H_out);
    out.push_back(W_out);
    return out;
}

Tensor conv2D::forward(const Tensor& input){
    throw std::runtime_error("forward of conv2D not implemented yet");
}

long long conv2D :: parameter_count() const{
    return (long long)in_channel*out_channel*(kernel_size*kernel_size) + out_channel;
    // area of one kernel * number of kernels
}