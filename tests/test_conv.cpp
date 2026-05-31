#include"Conv2D.hpp"
int main(){
    conv2D conv(3,16,3,1,1);
    std::cout<<conv.name()<<"\n";
    for(auto x:conv.output_shape({3,28,28})){
        std::cout<<x<<" ";
    }
    return 0;
}