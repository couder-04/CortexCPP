#include "ModelSummary.hpp"

#include "Sequential.hpp"
#include "Conv2D.hpp"
#include "BatchNorm2D.hpp"
#include "ReLU.hpp"
#include "MaxPool2D.hpp"
#include "Flatten.hpp"
#include "Linear.hpp"


int main(){
    try{
        class sequential model;
            std::cout<<"GOOD MODEL:\n";
            model.add(std::make_unique<conv2D>(3,16,3,1,1));// in, out, k_size, stride, padding
            // object type is conv2D and makeunique creates and returns a pointer for it
            model.add(std::make_unique<batchnorm2D>(16));
            model.add(std::make_unique<ReLU>());
            model.add(std::make_unique<maxpool2D>(2,2));// kernel, stride
            model.add(std::make_unique<Flatten>());
            model.add(std::make_unique<Linear>(4096,10));
            std::vector<int> input_shape ={3,32,32};
            model_summary:: summary(model, input_shape);


            std:: cout<<"BAD MODEL\n";
            sequential bad;
            bad.add(std:: make_unique<conv2D>(3,16,3,1,1));
            bad.add(std:: make_unique<Linear>(100,10));
            model_summary:: summary(bad, input_shape);

    }
    catch(const std :: exception &e){
        std::cout<<e.what()<<"\n";
    }
    
    return 0;
}