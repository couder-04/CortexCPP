//  make_unique makes a new pointer of that type and returns

#include "Sequential.hpp"
#include "Conv2D.hpp"
#include "MaxPool2D.hpp"
#include "Flatten.hpp"
#include "Linear.hpp"
#include "ShapeInference.hpp"


// void add(std::unique_ptr<Layer>); 
// adds a new layer object to the Vector of layers


int main(){
    sequential model;
    // make_unique<__LAYER_TYPE__> is a moveable ptr of the type LAYER_TYPE
    model.add(std::make_unique<conv2D>(3,16,3,1,1));
    model.add(std::make_unique<conv2D>(16,32,3,1,1));
    model.add(std::make_unique<maxpool2D>(2,2));
    model.add(std::make_unique<Flatten>());
    model.add(std::make_unique<Linear>(32*16*16,10));
    model.print_architecture();
    ShapeInfer::print_shapes(model, {3,28,28});
    return 0;
}