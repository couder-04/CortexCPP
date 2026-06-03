#include "Sequential.hpp"

//  we have a VECTOR LAYERS which is a vector of the layer objects

void sequential::add(Layer* layer){
    layers.push_back(layer);
}

sequential:: ~sequential(){     // destructor
    for(auto layer: layers){    // deletes the layer objects from heap one by one
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
        curr=layer->forward(curr);// curr is the Tensor that flows across the layers
    }
    return curr;
}