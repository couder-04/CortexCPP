#include "Tensor.hpp"       // header file included
#include <numeric>          //used for vector wide math ops
#include <functional>       // helps in function wrapping
#include <stdexcept>        // for throwing exceptions

Tensor:: Tensor(const std::vector <int>& input_shape):shape(input_shape)// initializer version
{// constructor class initialized
    if((int)input_shape.size()==0){
        throw std::runtime_error("Dimensions missing");
    }
    for(int i=0; i<(int)input_shape.size(); i++){
        if(input_shape[i]<=0){
            throw std::runtime_error("Dimension negative or zero");
        }
    }
    data.resize(numel());// allocates the memeory on its own to the data vector
    strides.resize((int)shape.size());// allocates the memeory on its own to the shape vector
    compute_strides();// strides computed and stored in private
}

Tensor:: Tensor(){
    // Empty Tensor initialized
    data={};
    shape={};
    strides={};
}

long long Tensor :: numel() const{// total number of elements based on shape 
    if(shape.empty()){return 0;}
    // const makes it a read only function i.e it cant alter the TENSOR object
    long long eles= std::accumulate(shape.begin(), shape.end(),1LL,std::multiplies<long long>());
    // start to end prod _init_ ==1LL and multiplies (from the functional lib) will use 2 long long and return their prod
    return eles;
}

void Tensor::compute_strides(){// Jumps computed
    long long curr=1;       // stride[i] = prod of all dim i+1 to n-1
    if(shape.empty()){return;}
    for(int i=(int)shape.size()-1; i>=0; i--){
        strides[i]=curr;
        curr *=shape[i];
    }
}

long long Tensor::compute_index(const std::vector <int> &indices )const{     // 1D index of the point in the tensor block
    long long idx=0;
    if((int)indices.size()!= (int)shape.size()){
        throw std::runtime_error("Dimension Mismatch");
    }
    for(int i=0; i<(int)indices.size(); i++){
        if(indices[i]<0 || indices[i]>=shape[i]){
            throw std::runtime_error("Invalid access");
        }
        idx+=indices[i]*strides[i];     // sum of (stride[i]*index[i])
    }
    return idx;
}

// FOR MUTABLE TENSORS
float& Tensor:: at(const std::vector<int> &indices){ //         const float& return a const reference to a float (another name for same memory location)
    return data[compute_index(indices)];
}
//      IMPORTANT
// float& creates/returns an alias to a float stored at some memory location
// NOT the memory location integer itself.


// FOR READ ONLY            ""     CONST TENSORS      ""
const float& Tensor::at(const std::vector<int>& indices) const{
    return data[compute_index(indices)];
}

void Tensor::reshape(const std::vector<int>&new_shape){     // can change shape but need to preserve the data
    if(new_shape.empty()){
        throw std::runtime_error("Empty reshape dimensions");
    }
    for(int x: new_shape){
        if(x<=0){
            throw std::runtime_error("Invalid reshape dimensions");
        }
    }
    long long old_ele=Tensor::numel();
    long long new_ele= std::accumulate(new_shape.begin(),new_shape.end(),1LL,std::multiplies<long long>());
    if(old_ele==new_ele){
        shape=new_shape;
        strides.resize(shape.size());
        compute_strides();}
    else{
        throw std::runtime_error("Invalid reshape dimensions");
    }
}

void Tensor::print_shape()const{ // print out the shape vector
    std::cout<<"(";
    for(int i=0; i<(int)shape.size(); i++){
        std::cout<<shape[i];
        if(i!=(int)shape.size()-1){std::cout<<",";}
    }
    std::cout<<")\n";
}

void Tensor::print_data()const{  // print out the 1D array in order
    for(int i=0; i<(int)data.size(); i++){
        std::cout<<data[i]<<" ";
    }
    std::cout<<"\n";
}

std::vector<float>& Tensor::raw_data(){
    return data;// returns a new referernce to the tensor data
}

const std::vector<float>& Tensor::raw_data() const{
    return data; // returns a new referernce to the tensor data
}

// this function just fetches you the shape and cant change the shape vector neither the input
const std::vector<int>& Tensor::get_shape() const{
    return shape;
}

// only reads object
// const T&

// modifies object
// T&

// needs independent copy
// T