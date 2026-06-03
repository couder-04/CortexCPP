#pragma once
#include"Layer.hpp"
#include<vector>
#include<string>

class batchnorm1D : public Layer{
    private:
        int num_features;
    public:
        batchnorm1D(int num_featuress);
        std:: string name() const override;
        Tensor forward(const Tensor& input)override;
        std::vector<int>output_shape(const std::vector<int>& input_shape) const override;
};