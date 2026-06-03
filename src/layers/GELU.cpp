#include "GELU.hpp"
#include  "math.h"
std::string GELU::name() const{// const means that it doesnt modeify the tensor
    return "GELU";
}

std::vector<int> GELU::output_shape(const std::vector<int> &input_shape)const{
    return input_shape;
}
// create output tensor->iterate through input values
// ->apply GeLU logic->store results in output->return output

Tensor GELU :: forward(const Tensor &input){
    Tensor output(input.get_shape());// output tensor instantiated 
    const auto& in_data= input.raw_data();// READ ONLY ALIAS OF THE INPUT VECTOR
    auto& out_data= output.raw_data(); // WRITEABLE ALIAS FOR DATA IN OUTPUT_DATA vector
    double pi = acos(-1);
    for(int i=0; i<(int)in_data.size(); i++){// gelu logic
        float x= in_data[i] ;
        out_data[i]=x* 0.5 *(1 + tanh(sqrt(2/pi)*(x+ 0.044715*pow(x,3))));
    }
    return output;
}

long long GELU:: parameter_count() const{
    return 0;
}
