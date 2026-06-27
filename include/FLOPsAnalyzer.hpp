//  we count the number of floating point operations per layer
#pragma once

#include "Sequential.hpp"

class FLOPs{
    public: 
        static long long total_FLOPs( sequential &model,  std:: vector <int> & input_shape);
        static void analyze( sequential& model, std::vector<int>& input_shape);
};