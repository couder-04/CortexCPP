#pragma once
#include "Layer.hpp"

class LeakyReLU: public Layer{ 
    private:
        float alpha;
    // publicly inherit Layer interface
    
    public:
        LeakyReLU(float alpha= 0.01f);
        Tensor forward(const Tensor& input)override;// retunrs a new transformed object
        std::vector<int> output_shape(const std::vector<int>&input_shape)const override;
        //     OVERRIDE MEANS:   "this function overrides a virtual parent function"
        std::string name() const override;
};