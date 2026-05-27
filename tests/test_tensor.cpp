#include "Tensor.hpp"

int main(){
    try{
        Tensor x({3,32,32});
        x.print_shape();
        std::cout<<x.numel()<<"\n";
        x.at({2,9,4})=6.986;
        std::cout<<x.at({2,9,4})<<"\n";
        x.reshape({3072});
        x.print_shape();
    }
    catch(std::runtime_error& e){
        std::cout<<"Error: "<<e.what()<<"\n";
    }
    return 0;
}

