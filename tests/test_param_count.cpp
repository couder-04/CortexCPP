#include "Sequential.hpp"
#include "Conv2D.hpp"
#include "ReLU.hpp"
#include "MaxPool2D.hpp"
#include "Flatten.hpp"
#include "Linear.hpp"
#include "BatchNorm2D.hpp"
#include "ParameterCounter.hpp"
#include <iostream>
#include <memory>

int main()
{
    sequential model;

    model.add(
        std::make_unique<conv2D>(3,16,3,1,1));

    model.add(std::make_unique<batchnorm2D>(16));

    model.add(std::make_unique<ReLU>());

    model.add(std::make_unique<maxpool2D>(2,2));

    model.add(std::make_unique<Flatten>());

    model.add(std::make_unique<Linear>(4096,10));
    std::cout<< "========================\n";

    std::cout<< " PARAMETER SUMMARY\n";

    std::cout<< "========================\n";

    param_count::print_summary(model);

    std::cout<< "\nTotal Parameters = "<< param_count::total_params(model)<< "\n";

    return 0;
}