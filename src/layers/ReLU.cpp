#include "ReLU.hpp"

std::string ReLU::name() const{// const means that it doesnt modeify the tensor
    return "RELU";
}

std::vector<int> ReLU::output_shape(const std::vector<int> &input_shape)const{
    return input_shape;
}
// create output tensor->iterate through input values
// ->apply ReLU logic->store results in output->return output

Tensor ReLU:: forward(const Tensor &input){
    Tensor output(input.get_shape());// output tensor instantiated 
    const auto& in_data= input.raw_data();// READ ONLY ALIAS OF THE INPUT VECTOR
    auto& out_data= output.raw_data(); // WRITEABLE ALIAS FOR DATA IN OUTPUT_DATA vector
    for(int i=0; i<(int)in_data.size(); i++){// RELU LOGIC
        if(in_data[i]<0){out_data[i]=0;}
        else{out_data[i]=in_data[i];}
    }
    return output;
}


