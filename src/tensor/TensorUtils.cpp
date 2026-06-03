#include "TensorUtils.hpp"
#include <numeric>          //used for vector wide math ops
#include <functional>       // helps in function wrapping
#include <limits>
#include<stdexcept> 
#include <random>

long long TensorUtils:: numel(const std:: vector<int>& input_shape){
    if(input_shape.empty()){return 0;}    
    long long eles= std::accumulate(input_shape.begin(), input_shape.end(),1LL,std::multiplies<long long>());
    return eles;
}

bool TensorUtils:: same_shape(const Tensor& a, const Tensor&b){
    if(a.get_shape()== b.get_shape()){
        return true;
    }
    return false;
}

std:: vector<int> TensorUtils::flatten_shape(const std:: vector<int>& shape){
    std::vector <int> out;
    int eles= std::accumulate(shape.begin(), shape.end(),1,std::multiplies<int>());
    out.push_back(eles);
    return out;
}

Tensor TensorUtils::zeros(const std::vector<int>& shape){
    Tensor out(shape);
    for(auto &x: out.raw_data()){// use the alias not the copy
        x=0;
    }
    return out;
}

Tensor TensorUtils::ones(const std::vector<int>& shape){
    Tensor out(shape);
    for(auto &x: out.raw_data()){
        x=1;
    }
    return out;
}


// Create tensor
// Create random engine
// Create distribution [0,1]
// Visit every tensor element
// Generate random number
// Store random number
// Return tensor


Tensor TensorUtils::random_uniform(const std::vector<int>& shape){
    Tensor out(shape);
    std:: random_device rd;// asks os
    std:: mt19937 gen(rd); // uses this mt19937 algo to generate the rabndom terms
    std:: uniform_int_distribution<float>dist(0.0f,1.0f);// distribution generated
    for(auto& x: out.raw_data()){
        x= dist(gen);   // from this distibution we generate  terms
    }
    return out;
}

std::string  TensorUtils:: shape_to_string(const std:: vector<int> & shape){
    std:: string seq;
    seq += '(';
    int i=0;
    for(auto x: shape){
        i++;
        seq+=std::to_string(x);
        if(i!=shape.size()){seq+=",";}
    }
    seq+=')';
    return seq;
}


float TensorUtils :: mean(const Tensor& t){
    float sum=0;
    float mean=0;
    for(auto x: t.raw_data()){
        sum+=x;
    }
    mean=sum/t.numel();
    return mean;
}


float TensorUtils :: var(const Tensor& t){
    float mean= TensorUtils::mean(t);
    float sum=0;
    for(auto x: t.raw_data()){
        sum+=(x-mean)*(x-mean);
    }
    return sum/t.numel();
}

float TensorUtils:: max(const Tensor& t){
    float maxi = std::numeric_limits<float>::lowest();
    for(auto x: t.raw_data()){
        if(x>maxi){maxi=x;}
    }
    return maxi;
}

float TensorUtils:: min(const Tensor& t){
    float mini = std::numeric_limits<float>::max();
    for(auto x: t.raw_data()){
        if(x<mini){mini=x;}
    }
    return mini;
}

void TensorUtils:: assert_same_shape(Tensor & a, Tensor& b){
    if(a.get_shape() != b.get_shape()){
        throw std:: runtime_error("Shape of tensors dont match");
    }
}
void TensorUtils:: assert_1D(const Tensor & a){
    if(a.get_shape().size()!=1){
        throw std:: runtime_error("tensor not 1D");
    }
}
void TensorUtils:: assert_2D(const Tensor & a){
    if(a.get_shape().size()!=2){
        throw std:: runtime_error("tensor not 2D");
    }
}
void TensorUtils:: assert_3D(const Tensor & a){
    if(a.get_shape().size()!=3){
        throw std:: runtime_error("tensor not 3D");
    }
}