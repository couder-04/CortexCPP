#pragma once
#include"Layer.hpp"
#include"Tensor.hpp"
class Linear : public Layer{        //sub class created
    private:
        int in_features;    //in_shape
        int out_features;   //out_shape
        Tensor weights;     //weights tensor
        Tensor bias;        //bias tensor
    public:
        Linear(int in_features, int out_features);      //  constructor
        Tensor forward(const Tensor& input) override;   //  output
        std::string name() const override;      //namee
        std::vector<int> output_shape(const std::vector<int>&input_shape) const override;
        Tensor& get_weights();
        long long parameter_count() const override;
        Tensor& get_bias();
        // output shape
};
// Linear stores objects (Tensor), 
// not heap-allocated pointers, 
// so the default destructor automatically cleans everything up.
//  => no destructor needed