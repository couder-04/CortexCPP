#pragma once
#include "Layer.hpp"
#include<string>
#include<vector>

class conv2D: public Layer{
    private:
        int in_channel;
        int out_channel;
        int kernel_size;
        int stride;
        int padding;
    public:
        conv2D(int in_channel,int out_channel,int kernel_size,int stride=1,int padding=0);
        std:: string name()const override;
        std:: vector<int> output_shape(const std:: vector<int >&input_shape)const override;
        Tensor forward(const Tensor& input) override;
        long long parameter_count() const override;
};