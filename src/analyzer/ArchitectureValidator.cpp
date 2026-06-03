// 1. validate()
// 2. ShapeInference exception handling
// 3. Conv after Linear
// 4. MaxPool after Linear
// 5. Linear after Conv without Flatten
// 6. Flatten after Softmax

#include "ArchitectureValidator.hpp"
#include "Sequential.hpp"
#include "ShapeInference.hpp"
bool ArchitectureValidator :: validate(sequential& model, std:: vector<int>& input_shape){
    try{
        ShapeInfer::infer(model, input_shape);
    }
    catch(const std:: exception &e){
        std::cout<<e.what()<<"\n";
        return false;
    }
    int tag_linear=0;
    int tag_conv2d=0;
    int tag_flatten=0;
    int tag_maxpool2D=0;
    int tag_maxpool1D=0;
    int tag_softmax=0;
    for(auto &x: model.get_layer()){
        if(x->name()=="flatten"){tag_flatten=1;}
        if(x->name()=="maxpool2D"){tag_maxpool2D=1;}
        if(x->name()=="maxpool1D"){tag_maxpool1D=1;}
        if(x->name()=="conv2D"){tag_conv2d=1;}
        if(x->name()=="linear"){tag_linear=1;}
        if(x->name()=="softmax"){tag_softmax=1;}

        if(x->name()=="conv2D" && tag_linear==1){ std::cout<< "Conv2D cannot follow Linear\n";return false;}
        if(x->name()=="maxpool1D" && tag_linear==1){std::cout<< "maxpool1D cannot follow Linear\n";return false;}
        if(x->name()=="maxpool2D" && tag_linear==1){std::cout<< "maxpool2D cannot follow Linear\n";return false;}
        if(x->name()=="linear" && tag_conv2d==1 && tag_flatten==0){std::cout<< "linear cannot follow conv2d without flattening\n";return false;}
        if(x->name()=="flatten" && tag_softmax==1){std::cout<< "linear cannot follow softmax\n";return false;}
    }
    return true;
}


