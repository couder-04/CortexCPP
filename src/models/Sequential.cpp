#include "Sequential.hpp"

void sequential::add(Layer* layer){
    layers.push_back(layer);
}

sequential:: ~sequential(){     // destructor
    for(auto layer: layers){    // deletes the layer from heap 
        delete layer;
    }
}

void sequential:: print_architecture()const{
    std::cout<<"Model Architecture\n";
    for(int i=0; i<(int)layers.size(); i++){
        std::cout<<i+1<<"."<<layers[i]->name()<<"\n";
    }
    // calling the name method for each object pointer
}

Tensor sequential::forward(const Tensor &input){
    Tensor curr=input;
    for(auto layer: layers){
        curr=layer->forward(curr);
    }
    return curr;
}