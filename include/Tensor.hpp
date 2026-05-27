//      .hpp file contrain the blueprint of the .cpp file

#pragma once

// # include            : is a std I/O preprocessor directive , which Converts the code in to a raw c++ file stripping off the comments before compilation
#include <iostream> // iostream file :  is a standard library header file that provides the core functionality for managing program input and output
#include<vector> // header for vector

class Tensor{
    private:
        std::vector<float> data; // data is stored as floats
        std::vector<int> shape;  // shape of the tensor
        std::vector<long long> strides; // for jumping from one c->c , r->r, channel->chanel etc
    public:
        Tensor();
        Tensor(const std::vector<int> &shape); //constructor of the tensor class: initializes the class properly
        long long numel() const ; // number of elements
        void print_data() const; // data of the tensor
        void print_shape() const; // shape of the tensor
        void compute_strides() ;// calculating the jumps for each dimension
        long long compute_index(const std::vector<int> &indices) const;  //calculating the index in 1d array of that elment 
        float& at (const std::vector<int> &indices);
        const float& at(const std::vector<int>& indices) const;
        void reshape(const std::vector<int> & new_shape);
};


// #ifndef MY_HEADER_H: Checks if this unique name is not defined yet.
// #define MY_HEADER_H: Immediately defines the name so future checks fail.
// #endif: Marks the end of the protected code block.

// alll replaced for once by PRAGMA
