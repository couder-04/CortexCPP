#include "ParameterCounter.hpp"

long long param_count:: total_params(const sequential &model){
    long long total =0;
    for(auto &x : model.get_layer()){
        total+=x->parameter_count();
    }
    return total;
}
void param_count :: print_summary(const sequential &model){
    for(auto & x: model.get_layer()){
        std:: cout<<x->name()<<" : ";
        std:: cout<<x->parameter_count()<<"\n";
    }
    std::cout<<"Total params:"<<total_params(model)<<"\n";
}

// In C++, std::make_unique is a factory function that allocates
//  memory for an object on the heap and wraps it inside a std::unique_ptr. 