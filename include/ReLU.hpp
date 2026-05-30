#pragma once
#include "Layer.hpp"

class ReLU: public Layer{ // Relu inherits from the Layer
    // publicly inherit Layer interface
    // Layer* x = new ReLU(); is valid now
    public:
        Tensor forward(const Tensor& input)override;// retunrs a new transformed object
        std::vector<int> output_shape(const std::vector<int>&input_shape)const override;
        //     OVERRIDE MEANS:   "this function overrides a virtual parent function"
        std::string name() const override;
};