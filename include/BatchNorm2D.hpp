#pragma once
#include"Layer.hpp"
#include<vector>
#include<string>

class batchnorm2D : public Layer{
    private:
        int num_features;
        // in the input layer num_features reqd to:
        // BatchNorm keeps the numbers flowing 
        // through the network in a reasonable range so training becomes
        //  faster and more stable. 
        // gamma = scale
        // beta  = shift
        // output =  gamma* normalized value + beta
        // LINEAR LAYER:  Creates new information by mixing features.
        // BATCH NORM: Keeps existing features healthy and well-scaled.


    public:
        batchnorm2D(int num_featuress);
        std:: string name() const override;
        Tensor forward(const Tensor& input)override;
        std::vector<int>output_shape(const std::vector<int>& input_shape) const override;
        long long parameter_count() const override;
};