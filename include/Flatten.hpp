#pragma once
#include "Layer.hpp"
#include <numeric>
#include <functional>
class Flatten : public Layer{
    public:
        Tensor forward(const Tensor& input) override;
        // const in the parameter protects the INPUT SHAPE
        //  const outside says that this function cant MODIFY the Flatten object
        std::vector<int> output_shape(const std:: vector<int> &input_shape)const override;
        long long parameter_count() const override;
        std::string name() const override;
};
