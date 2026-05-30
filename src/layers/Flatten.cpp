#include "Flatten.hpp"

// create output tensor ->store results in output ->return output
Tensor Flatten::forward(const Tensor &input) {   
    Tensor output(output_shape(input.get_shape()));
    const auto& in_data= input.raw_data();// READ ONLY ALIAS OF THE INPUT VECTOR
    auto& out_data= output.raw_data(); // WRITEABLE ALIAS FOR DATA IN OUTPUT_DATA vector
    for(int i=0; i<(int)in_data.size(); i++){// flattened layer
        out_data[i]=in_data[i];
    }
    return output;
}

std::vector<int> Flatten ::output_shape(const std::vector<int> &input_shape) const
{
    std::vector<int> out;
    out.resize(1);
    out[0]=std::accumulate(input_shape.begin(), input_shape.end(), 1LL, std::multiplies<long long>());
    return out;
}

std:: string Flatten :: name() const {
    return "FLATTEN";
}


