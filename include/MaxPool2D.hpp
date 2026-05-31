#pragma once
#include<vector>
#include<string>
#include "Layer.hpp"
class maxpool2D: public Layer {
    private:
        int kernel_size;
        int stride;
    public:
        maxpool2D(int kernel_size, int stide);
        std:: string name()const override;
        std::vector<int>output_shape(const std::vector<int>& input_shape)const override;
        Tensor forward(const Tensor& input)override;
};