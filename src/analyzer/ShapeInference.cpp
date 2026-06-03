// 1. Get layers from Sequential
// 2. Start with input_shape
// 3. Apply output_shape() repeatedly
// 4. Store every shape
// 5. Return all shapes
// 6. Print helper



// std::unique_ptr is a smart pointer in C++ that retains exclusive ownership of a dynamically allocated object 
// and automatically deallocates that object when the pointer goes out of scope.


#include "ShapeInference.hpp"
# include"TensorUtils.hpp"

std::vector<std::vector<int>> ShapeInfer:: infer(const sequential &model, const std::vector<int> &input_shape){
    std::vector<std::vector<int>> out;
    std:: vector <int> curr= input_shape;
    out.push_back(curr);
    for(const auto& x :model.get_layer()){    // using get_layer() we get the layers in the alias form
        curr=x->output_shape(curr);
        out.push_back(curr);
    }
    return out;
}

void ShapeInfer:: print_shapes(const sequential &model, const std::vector<int> &input_shape){
    std::vector<std::vector<int>> out= ShapeInfer::infer(model,input_shape);
    for(auto x: out){
        std::cout<<TensorUtils::shape_to_string(x)<<"\n";
    }
}


