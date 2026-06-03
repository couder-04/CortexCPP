#pragma once
#include"Layer.hpp"
#include<vector>
#include<string>

class dropout : public Layer{
    private:
        float probab;
        // Dropout randomly “turns off” some neurons during training so the 
        // network doesn’t become overly dependent on a few neurons.


    public:
        dropout(float probab);
        std:: string name() const override;
        Tensor forward(const Tensor& input)override;
        std::vector<int>output_shape(const std::vector<int>& input_shape) const override;
        long long parameter_count() const override;
};