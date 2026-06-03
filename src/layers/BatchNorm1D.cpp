#include "Layer.hpp"
#include "BatchNorm1D.hpp"

batchnorm1D:: batchnorm1D(int num_features): num_features(num_features){}

std:: string batchnorm1D:: name() const {
    return "batchnorm1D";
}

std:: vector <int> batchnorm1D:: output_shape(const std:: vector <int> & input_shape) const{
    if(input_shape[0]!= num_features){throw std:: runtime_error("incorrect num_features");}
    else if(input_shape.size()!=3) {throw std:: runtime_error("incorrect input_shape dimension");}
    else{return input_shape;}
}

Tensor batchnorm1D::forward(const Tensor& input) {
    throw std:: runtime_error("fwd not implementted");
}
