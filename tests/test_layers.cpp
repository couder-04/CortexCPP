#include "Tensor.hpp"
#include "ReLU.hpp"
#include "Flatten.hpp"
#include  "Linear.hpp"
#include "Softmax.hpp"
int main(){
    Tensor x({4,3});// tensor object x created

    for(int i=0; i<4; i++){
        for(int j=0; j<3; j++){
            x.at({i,j})= -i+2*j;// filled data inside the tensor object
        }
    }

    ReLU relu;// layer object
    // FWD PASS
    Tensor y= relu.forward(x);
    x.print_data();
    y.print_data();
    x.print_shape();
    y.print_shape();    
    Flatten f;
    Tensor z= f.forward(y);
    std::cout<<f.name()<<"\n";
    z.print_data();
    z.print_shape();
    x.print_shape();
    std::cout<<relu.name()<<"\n";
    y.print_shape();
    z.print_shape();
    //z is a flat vector of 24 f_in
    int f_in =z.get_shape()[0];
    int f_out= 10;
    Linear lin(f_in,f_out);
    // Weights:
    for(int i=0; i<f_out; i++){
        for (int j = 0; j < f_in; j++){
            lin.get_weights().at({i,j})=i-4*j;
        }
    }
    for (int i = 0; i < f_out; i++){
        lin.get_bias().at({i})=i-6;
    }
    std::cout<<lin.name()<<"\n";
    std::cout<<"WEIGHTS:\n";
    lin.get_weights().print_data();
    std::cout<<"BIAS:\n";
    lin.get_bias().print_data();
    std::cout<<"OUTPUT shape\n";
    lin.output_shape(z.get_shape());
    std::cout<<"Result\n";
    Tensor result =lin.forward(z);
    result.print_data();
    std::cout<<"Number of elements\n";
    std::cout<<result.numel()<<"\n";
    Linear fc(2,2);

    // Weights
    fc.get_weights().at({0,0}) = 1;
    fc.get_weights().at({0,1}) = 2;

    fc.get_weights().at({1,0}) = 3;
    fc.get_weights().at({1,1}) = 4;

    // Bias
    fc.get_bias().at({0}) = 10;
    fc.get_bias().at({1}) = 20;

    // Input
    Tensor q({2});

    q.at({0}) = 5;
    q.at({1}) = 6;
    Tensor out= fc.forward(q);
    out.print_shape();
    out.print_data();

    std::cout<<"before softmax:\n";
    z.print_data();
    std::cout<<"after softmax:\n";
    softmax soft;
    std::cout<<soft.name()<<"\n";
    std::cout<<"Output_shape:\n";
    std::cout<<soft.output_shape(z.get_shape())[0]<<"\n";
    soft.forward(z).print_data();
}


// RELU WORKING PERFECT
// Flatten WORKING PERFECT
// Linear Layer working PERFECT
// softmax working PERFECT
// CONV2D: fwd left
// maxpool2D: fwd left