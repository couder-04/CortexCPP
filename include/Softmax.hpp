#pragma once
#include"Layer.hpp"
#include<vector>
#include<string>

class softmax : public Layer{
    public:
        std:: string name() const override;
        Tensor forward(const Tensor& input)override;
        std::vector<int>output_shape(const std::vector<int>& input_shape) const override;
};