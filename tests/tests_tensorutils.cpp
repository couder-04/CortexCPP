#include "Tensor.hpp"
#include "TensorUtils.hpp"
#include <iostream>

int main() {

    std::cout << "numel:\n";
    std::cout << TensorUtils::numel({3,4,2}) << "\n\n";

    std::cout << "flatten_shape:\n";

    auto flat = TensorUtils::flatten_shape({3,4,2});

    for(auto x : flat){
        std::cout << x << " ";
    }

    std::cout << "\n\n";

    std::cout << "zeros:\n";

    Tensor z = TensorUtils::zeros({2,3});

    z.print_shape();
    z.print_data();

    std::cout << "\n";

    std::cout << "ones:\n";

    Tensor o = TensorUtils::ones({2,3});

    o.print_shape();
    o.print_data();

    std::cout << "\n";

    std::cout << "shape_to_string:\n";

    std::cout
        << TensorUtils::shape_to_string({3,32,32})
        << "\n\n";

    Tensor t({5});

    t.at({0}) = 1;
    t.at({1}) = 2;
    t.at({2}) = 3;
    t.at({3}) = 4;
    t.at({4}) = 5;

    std::cout << "mean:\n";
    std::cout << TensorUtils::mean(t) << "\n\n";

    std::cout << "variance:\n";
    std::cout << TensorUtils::var(t) << "\n\n";

    std::cout << "max:\n";
    std::cout << TensorUtils::max(t) << "\n\n";

    std::cout << "min:\n";
    std::cout << TensorUtils::min(t) << "\n\n";

    Tensor a({2,3});
    Tensor b({2,3});
    Tensor c({3,2});

    std::cout << "same_shape:\n";

    std::cout
        << TensorUtils::same_shape(a,b)
        << "\n";

    std::cout
        << TensorUtils::same_shape(a,c)
        << "\n\n";

    std::cout << "random_uniform:\n";

    Tensor r =TensorUtils::random_uniform({2,3});

    r.print_shape();
    r.print_data();

    std::cout << "\n";

    return 0;
}