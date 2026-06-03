#pragma once
#include "Layer.hpp"

class sigmoid: public Layer{ 
    // publicly inherit Layer interface
    
    public:
        Tensor forward(const Tensor& input)override;// retunrs a new transformed object
        std::vector<int> output_shape(const std::vector<int>&input_shape)const override;
        //     OVERRIDE MEANS:   "this function overrides a virtual parent function"
        std::string name() const override;
        long long parameter_count() const override;
};