#include "Sequential.hpp"

namespace param_count{
    // returns a vector of shape vectors
    long long total_params(const sequential &model);
    void print_summary(const sequential &model);
}