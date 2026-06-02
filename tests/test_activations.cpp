# include "Layer.hpp"
# include "GELU.hpp"
# include "ReLU.hpp"
# include "GELU.hpp"
# include "LeakyReLU.hpp"
# include "Sigmoid.hpp"
# include "Tanh.hpp"

int main(){
    Tensor x({5});
    x.at({0})=-2;
    x.at({1})=-1;
    x.at({2})=0;
    x.at({3})=1;
    x.at({4})=2;

    ReLU relu;
    LeakyReLU lrelu;
    GELU gelu;
    sigmoid sig;
    TanH TANH;

    std::cout<<"Numer of elements:\n";
    std::cout<<x.numel()<<"\n";
    std::cout<<"Shape of Tensor:\n";
    x.print_shape();
    std::cout<<"Data:\n";
    x.print_data();

    std::cout<<relu.name()<<"\n";
    relu.forward(x).print_data();
    std::cout<<lrelu.name()<<"\n";
    lrelu.forward(x).print_data();
    std::cout<<gelu.name()<<"\n";
    gelu.forward(x).print_data();
    std::cout<<sig.name()<<"\n";
    sig.forward(x).print_data();
    std::cout<<TANH.name()<<"\n";
    TANH.forward(x).print_data();

    return 0;
}