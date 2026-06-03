#include "Sigmoid.hpp"

std::string sigmoid::name() const{// const means that it doesnt modeify the tensor
    return "sigmoid";
}

std::vector<int> sigmoid::output_shape(const std::vector<int> &input_shape)const{
    return input_shape;
}
// create output tensor->iterate through input values
// ->apply sigmoid logic->store results in output->return output

Tensor sigmoid:: forward(const Tensor &input){
    Tensor output(input.get_shape());// output tensor instantiated 
    const auto& in_data= input.raw_data();// READ ONLY ALIAS OF THE INPUT VECTOR
    auto& out_data= output.raw_data(); // WRITEABLE ALIAS FOR DATA IN OUTPUT_DATA vector
    for(int i=0; i<(int)in_data.size(); i++){// sigmoid LOGIC
        out_data[i]=1/(1+exp(-1*in_data[i]));
    }
    return output;
}

long long sigmoid:: parameter_count() const{
    return 0;
}
