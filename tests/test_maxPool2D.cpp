#include"MaxPool2D.hpp"
int main(){
    maxpool2D Pool(2,2);
    for(auto x : Pool.output_shape({16,32,32})){

    std::cout << x << " ";

    }   

    std::cout << "\n";

    maxpool2D pool2(3,2);

    for(auto x : pool2.output_shape({16,32,32})){

        std::cout << x << " ";

    }

    std::cout << "\n";
    return 0;
}