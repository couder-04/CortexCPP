#include "Sequential.hpp"
#include  "Flatten.hpp"
#include "ReLU.hpp"
// Flatten and ReLU are sub classes of the class LAYER

int main(){
    Tensor x({3,4,2});
    std:: vector<float> &data= x.raw_data();
    for(int i=0; i<x.numel(); i++){
        data[i]=-5+i;
    }   
    sequential model;
    // object Flatten created and added, its pointer is passed
    model.add(new Flatten());
    model.add(new ReLU());
    model.print_architecture();
    Tensor y= model.forward(x);
    std::cout<<"Input shape\n";
    x.print_shape();
    std::cout<<"Input data\n";
    x.print_data();
    std::cout<<"Output shape\n";
    y.print_shape();
    std::cout<<"Output data\n";
    y.print_data();
    return 0;
}


