#pragma once
#include "Tensor.hpp"
#include<string>

class Layer{
    public:
//" virtual " means Child classes can Override this function i.e diff layers can have diff fwd passes
//  = 0 means that its an abstract class now : 
// An abstract class is a restricted blueprint that cannot be instantiated directly. It acts as a base class, designed to be inherited

        virtual Tensor forward (const Tensor &input)= 0;// fwd pass templaye
        virtual std:: vector<int> out_shape(const std::vector <int>&input_shape)=0;// calculates the shape of output tensor
        virtual std:: string name()=0;// architecture printing
        virtual ~ Layer() = default;
// When deleting derived neural network layers through Layer pointers, destroy them correctly and safely.
};

