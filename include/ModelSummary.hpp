#pragma once
#include "Sequential.hpp"
#include "ParameterCounter.hpp"
#include "ShapeInference.hpp"
#include "ArchitectureValidator.hpp"
class sequential;

namespace model_summary{
    void summary(sequential &model, std::vector<int>&input_shape);
};

