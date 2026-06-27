#include "FLOPsAnalyzer.hpp"
#include "Linear.hpp"
#include "ReLU.hpp"
#include "MaxPool2D.hpp"
#include "Conv2D.hpp"
#include "ShapeInference.hpp"
#include "BatchNorm1D.hpp"
#include "BatchNorm2D.hpp"
#include "Softmax.hpp"

int conv_flops=0;
int relu_flops=0;
int pool_flops=0;
int linear_flops=0;
int batchnorm_flops=0;
int bn_1d_flops=0;
int bn_2d_flops=0;
int sftmax_flops=0;
int total =0;

void FLOPs::analyze( sequential& model, std::vector<int>& input_shape){
    const auto& shapes = ShapeInfer::infer(static_cast<const sequential&>(model), static_cast<const std::vector<int>&>(input_shape));
    for(int i=0; i<model.get_layer().size(); i++){

        auto &layer=model.get_layer()[i];
        auto out_shape= shapes[i+1];
        if(layer->name()=="conv2d"){
            auto conv = dynamic_cast<conv2D*>(layer.get());
            int cin=conv->get_in_channels();
            int cout=conv->get_out_channels();
            int k=conv->get_kernel_size();
            int h_out=out_shape[1];
            int w_out=out_shape[2];
            conv_flops+=2 * k * k * h_out * w_out; 
        }
        if(layer->name()=="relu"){
            auto relu = dynamic_cast<ReLU*>(layer.get());
            std:: vector<int> out= out_shape;
            int prod=1;
            for(int i=0; i<3; i++){
                prod*=out[i];
            }
            relu_flops += prod;
        }   

        if(layer->name()=="softmax"){
            auto sfmax = dynamic_cast<softmax*>(layer.get());
            // TODO: compute softmax FLOPs
            (void)sfmax;
        }

        if(layer->name()=="batchnorm1d"){
            auto bn = dynamic_cast<batchnorm1D*>(layer.get());
            bn_1d_flops += bn->get_features() * 2;
        }

        if(layer->name()=="maxpool2d"){
            auto pool = dynamic_cast<maxpool2D*>(layer.get());
            // TODO: compute maxpool2d FLOPs
            (void)pool;
        }
        if(layer->name()=="linear"){
            auto linear = dynamic_cast<Linear*>(layer.get());
            int in_features=linear->get_in_features();
            int out_features=linear->get_out_features();
            linear_flops+=2*in_features*out_features;
        }

    }
}