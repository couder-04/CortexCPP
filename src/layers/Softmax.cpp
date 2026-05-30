# include "Softmax.hpp"
#include <limits>
std::string softmax::name()const{
    return "softmax";
}

std::vector<int> softmax ::output_shape(const std:: vector<int>& input_shape)const{
    return input_shape;
}

Tensor softmax::forward(const Tensor& input){
    Tensor output(input.get_shape());
    float sum=0;
    float max=std::numeric_limits<float>::lowest();
    for(auto x: input.raw_data()){
        if(x>max){max=x;}
    }
    for(auto x: input.raw_data()){
        sum+=exp(x-max);
    }
    for(int i=0; i<(int)input.numel(); i++){
        output.raw_data()[i]=exp(input.raw_data()[i]-max)/sum;
    }
    return output;
}