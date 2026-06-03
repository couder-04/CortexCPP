// 1. validate()
// 2. ShapeInference exception handling
// 3. Conv after Linear
// 4. MaxPool after Linear
// 5. Linear after Conv without Flatten
// 6. Flatten after Softmax

#include "ArchitectureValidator.hpp"
#include "Sequential.hpp"
#include "ShapeInference.hpp"
bool ArchitectureValidator :: validate(sequential model, std:: vector<int>& input_shape){
    try{
        ShapeInfer::infer(model, input_shape);
    }
    catch(const std:: runtime_error &e){
        std::cout<<e.what()<<"\n";
        return false;
    }
    return true;
}


