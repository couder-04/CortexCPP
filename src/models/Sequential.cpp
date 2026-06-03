#include "Sequential.hpp"

//  we have a VECTOR LAYERS which is a vector of the layer objects

void sequential::add(std::unique_ptr<Layer>layer){
    layers.push_back(std::move(layer));
    //  std::unique_ptr is move-only.
}

sequential:: ~sequential(){     // destructor
    // destructor automatically called for unique_ptr objects
}

void sequential:: print_architecture()const{
    std::cout<<"Model Architecture\n";
    for(int i=0; i<(int)layers.size(); i++){
        std::cout<<i+1<<"."<<layers[i]->name()<<"\n";
    }
    // calling the name method for each object pointer
}

const std::vector<std::unique_ptr<Layer>>& sequential::get_layer() const {
    return layers;
}

Tensor sequential::forward(const Tensor &input){
    Tensor curr=input;
    for(auto& layer: layers){
        curr=layer->forward(curr);// curr is the Tensor that flows across the layers
    }
    return curr;
}
