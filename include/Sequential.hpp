# pragma once

#include "Layer.hpp"
#include <vector>
#include<iostream>
// concatenated Sum of all Layers is the SEQUENTIAL BLOCK
// sequential object is a kinda array of layers of diff types
class sequential{
    private:
        std::vector<Layer*> layers;
    public:
        void add(Layer* layer); // adds a new layer object to the Vector of layers
        void print_architecture()const;
        Tensor forward(const Tensor& input);
        ~sequential();
        // CANT USE DEFAULT AS DEFAULT WONT DESTROY LAYER OBJECTS just the pointers
        // ~Sequential() = default only destroys the stored pointers, 
        // not the heap-allocated layer objects they point to.
};