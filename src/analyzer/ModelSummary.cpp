#include "ModelSummary.hpp"
#include <iomanip>
#include "TensorUtils.hpp"

// In C++, std::setw stands for set width and is a stream manipulator
// used to set the minimum character width of the next input or output operation. 
// It is primarily used to align data into clean columns or tables when printing text to the console

void model_summary::summary(sequential &model, std:: vector<int> & input_shape){
std::cout << "=================================\n";
std::cout << "MODEL SUMMARY\n";
std::cout << "=================================\n\n";

if(ArchitectureValidator::validate(model, input_shape)){std::cout<<"VALID ARCHITECTURE : YES";}
else{std::cout<<"VALID ARCHITECTURE : NO";return;}

std::cout<<"\n\n";
std::cout<< std::left<< std::setw(20) << "Layer"<< std::setw(20) << "Output Shape"<< std::setw(10) << "Params"<< "\n\n";

auto shapes =ShapeInfer::infer(model,input_shape);// set of all shapes

for(int i=0;i<model.get_layer().size();i++){
    const auto& layer =model.get_layer()[i];
    std::cout<< std::setw(20)<< layer->name()<< std::setw(20)<< TensorUtils::shape_to_string(shapes[i+1])<< std::setw(10)<< layer->parameter_count()<< "\n";
}

std::cout << "\n=================================\n\n";
std::cout<< "Total Parameters : "<< param_count::total_params(model)<< "\n\n";
std::cout << "=================================\n";
}