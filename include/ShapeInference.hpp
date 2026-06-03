#include "Sequential.hpp"

namespace ShapeInfer{
    // returns a vector of shape vectors
    std::vector<std::vector<int>> infer(const sequential &model, const std::vector<int> &input_shape);
    void print_shapes(const sequential &model, const std::vector<int> &input_shape);
}