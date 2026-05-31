#pragma once
#include"Layer.hpp"
#include<vector>
#include<string>

class batchnorm2D : public Layer{
    private:
        int num_features;
        // in the input layer num_features reqd to 
    public:
        batchnorm2D();
        std:: string name() const override;
        Tensor forward(const Tensor& input)override;
        std::vector<int>output_shape(const std::vector<int>& input_shape) const override;
};