#include "Tensor.hpp"
#include "ReLU.hpp"
#include "Flatten.hpp"
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
    if(z.numel()==y.numel()){std::cout<<"YO\n";}
}


// RELU WORKING PERFECT
// Flatten WORKING PERFECT