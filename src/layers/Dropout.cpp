#include "Layer.hpp"
#include "Dropout.hpp"

dropout:: dropout(float probab): probab(probab){}

std:: string dropout:: name() const {
    return "dropout";
}

std:: vector <int> dropout:: output_shape(const std:: vector <int> & input_shape) const{
    return input_shape;
}

Tensor dropout::forward(const Tensor& input) {
    throw std:: runtime_error("fwd not implementted");
}
