#pragma once
#include "Tensor.hpp"

// DEFINED A NAMESPACE

// In C++, a namespace is a declarative region used to organize code into logical groups and prevent naming conflicts.
//  It allows you to group identifiers like variables, functions, and classes under a specific name, 
// ensuring that names do not collide when integrating multiple libraries.


namespace TensorUtils{
    long long numel(const std:: vector<int>& input_shape);
    bool same_shape(const Tensor& a, const Tensor&b);
    std:: vector<int> flatten_shape(const std:: vector<int>& shape);

    Tensor zeros(const std::vector<int>& shape);
    Tensor ones(const std::vector<int>& shape);
    Tensor random_uniform(const std::vector<int>& shape);

    std::string shape_to_string(const std:: vector<int> & shape);

    float mean(const Tensor& t);
    float var(const Tensor& t);
    float max(const Tensor& t);
    float min(const Tensor& t);

    void assert_same_shape(Tensor & a, Tensor& b);
    void assert_1D(const Tensor & a);
    void assert_2D(const Tensor & a);
    void assert_3D(const Tensor & a);
}