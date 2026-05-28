#pragma once
#include "Layer.hpp"

class ReLU: public Layer{
    public:
        Tensor forward(const Tensor& input)override;
        std::vector<int> out_shape(const std::vector<int>&input_shape)const override;
};