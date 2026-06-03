#include "LeakyReLU.hpp"

LeakyReLU:: LeakyReLU(float alpha): alpha(alpha){}

std::string LeakyReLU::name() const{// const means that it doesnt modeify the tensor
    return "LeakyReLU";
}

std::vector<int> LeakyReLU::output_shape(const std::vector<int> &input_shape)const{
    return input_shape;
}
// create output tensor->iterate through input values
// ->apply Leaky_RELU logic->store results in output->return output

Tensor LeakyReLU:: forward(const Tensor &input){
    Tensor output(input.get_shape());// output tensor instantiated 
    const auto& in_data= input.raw_data();// READ ONLY ALIAS OF THE INPUT VECTOR
    auto& out_data= output.raw_data(); // WRITEABLE ALIAS FOR DATA IN OUTPUT_DATA vector
    for(int i=0; i<(int)in_data.size(); i++){// leaky RELU LOGIC
        if(in_data[i]<0){out_data[i]=in_data[i]*alpha;}
        else{out_data[i]=in_data[i];}
    }
    return output;
}


long long LeakyReLU :: parameter_count() const{
    return 0;
}
