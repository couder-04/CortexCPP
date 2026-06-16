#include "LayerFactory.hpp"
#include <string>

int main(){
    

    auto layer1= LayerFactory::create("relu");
    std:: cout<<layer1->name()<<"\n";
    auto layer2= LayerFactory::create("flatten");
    std:: cout<<layer2->name()<<"\n";
    auto layer3= LayerFactory::create("gelu");
    std:: cout<<layer3->name()<<"\n";
    auto layer4= LayerFactory::create("sigmoid");
    std:: cout<<layer4->name()<<"\n";
    auto layer5= LayerFactory::create("dropout 0.3");
    std:: cout<<layer5->name()<<"\n";
    auto layer6= LayerFactory::create("softmax");
    std:: cout<<layer6->name()<<"\n";    
    auto layer7= LayerFactory::create("linear 128 10");
    std:: cout<<layer7->name()<<"\n";    
    auto layer8= LayerFactory::create("conv2d 3 16 3 1 1");
    std:: cout<<layer8->name()<<"\n";    
    auto layer9= LayerFactory::create("maxpool 2 2");
    std:: cout<<layer9->name()<<"\n";    
    auto layer10= LayerFactory::create("batchnorm2d 16");
    std:: cout<<layer10->name()<<"\n";    
    auto layer11= LayerFactory::create("batchnorm1d 16");
    std:: cout<<layer11->name()<<"\n";    
    auto layer12= LayerFactory::create("leakyrelu 0.5");
    std:: cout<<layer12->name()<<"\n";    

    try{
        auto bad_layer= LayerFactory::create("apple");
    }
    catch(const std:: exception &e){
        std:: cout<< e.what()<<"\n";
    }
    return 0;
}