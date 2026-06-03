#include "Layer.hpp"
#include "BatchNorm2D.hpp"

batchnorm2D:: batchnorm2D(int num_features): num_features(num_features){}

std:: string batchnorm2D:: name() const {
    return "batchnorm2D";
}

std:: vector <int> batchnorm2D:: output_shape(const std:: vector <int> & input_shape) const{
    if(input_shape[0]!= num_features){throw std:: runtime_error("incorrect num_features");}
    else if(input_shape.size()!=3) {throw std:: runtime_error("incorrect input_shape dimension not 3D");}
    else{return input_shape;}
}

Tensor batchnorm2D::forward(const Tensor& input) {
    throw std:: runtime_error("fwd not implementted");
}

long long batchnorm2D :: parameter_count() const{
    return 2LL * num_features;
}