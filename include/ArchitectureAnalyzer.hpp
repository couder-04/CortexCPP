/*
input: 

Conv2D
ReLU
MaxPool2D
Flatten
Linear


output::

Architecture Score : 85/100

Suggestions:
- Add BatchNorm after Conv2D
- Consider Dropout before final Linear

Warnings:
- Large classifier head detected

Bottlenecks:
- Linear layer dominates parameter count


*/



#pragma once
#include "Sequential.hpp"
#include <string>
#include<vector>

class ArchitectureAnalyzer{
    public:
        static int score(sequential & model,  std:: vector<int> &input_shape);
        static std:: vector<std:: string> suggestions(const sequential & model);
        static std:: vector<std:: string> warnings(const sequential & model);
        //  function to call to print the analysis
        static void analyze( sequential& model,  std :: vector<int> &input_shape);
};