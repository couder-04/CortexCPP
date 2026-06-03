/*
y1 = w11*x1 + w12*x2 + w13*x3 + w14*x4 + b1
y2 = w21*x1 + w22*x2 + w23*x3 + w24*x4 + b2
y3 = w31*x1 + w32*x2 + w33*x3 + w34*x4 + b3         */ 

    //  Y=  W * X + B
    // (fout ,1)= (fout , fin)*(fin ,1) + (fout,1)

# include "Linear.hpp"
#include <stdexcept>
// constructor
Linear::Linear(int in_features,int out_features):
in_features(in_features),       //  initialized with in
out_features(out_features),     //  initialized with out
weights({out_features,in_features}),  //  initialized with shape (out, in)
bias({out_features})    //  initialized with shape (out)
{}

std::vector<int> Linear ::output_shape(const std::vector<int>&input_shape) const {
    if(input_shape.size()!=1){
        throw std::runtime_error("Linear expects 1d input");
    }
    return {out_features};
}

std::string Linear:: name() const {
    return "Linear";
}      

 Tensor& Linear:: get_weights()  {
    return weights;
}

 Tensor& Linear:: get_bias()  {
    return bias;
}

Tensor Linear:: forward(const Tensor& input){
    if(input.get_shape().size()!=1){
        throw std::runtime_error("Linear expects 1d input");
    }
    if(input.get_shape()[0]!=in_features){
        throw std:: runtime_error("Input dim mis_match");
    }
    Tensor out({out_features});
    for (int i = 0; i < out_features; i++){// choose a row in W
        float sum=bias.raw_data()[i];
        for (int j = 0; j < in_features; j++){    
            sum+=weights.at({i,j})*input.raw_data()[j];
            // element wise mutiplication of the row [j] with X[j]
        }
        out.at({i})=sum;
    }
    return out;
}       

long long Linear :: parameter_count() const{
    return (long long)in_features*out_features+ out_features;
}
