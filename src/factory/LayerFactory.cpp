#include "LayerFactory.hpp"

#include "ReLU.hpp"
#include "LeakyReLU.hpp"
#include "Sigmoid.hpp"
#include "Tanh.hpp"
#include "GELU.hpp"
#include "LeakyReLU.hpp"

#include "Flatten.hpp"
#include "Dropout.hpp"
#include "Softmax.hpp"

#include "BatchNorm1D.hpp"
#include "BatchNorm2D.hpp"

#include "Linear.hpp"

#include "Conv2D.hpp"
#include "MaxPool2D.hpp"

#include <sstream>
#include <stdexcept>


// initialize the stream with the string
// In C++, std::stringstream is a stream class from the <sstream> header 
// that allows you to treat a string object as a stream
std:: unique_ptr<Layer> LayerFactory ::create(const std:: string & cmd){

    //  PARSE OP
    std:: stringstream ss (cmd);
    std:: string op;
    ss>>op;// gets the first part of string in op


    if(op=="relu"){
        return std:: make_unique<ReLU>();
    }
    if(op=="flatten"){
        return std:: make_unique<Flatten>();
    }
    if(op=="softmax"){
        return std:: make_unique<softmax>();
    }
    if(op=="gelu"){
        return std:: make_unique<GELU>();
    }
    if(op=="sigmoid"){
        return std:: make_unique<sigmoid>();
    }
    if(op=="tanh"){
        return std:: make_unique<TanH>();
    }
    if(op=="dropout"){
        float p;
        ss>>p;
        return std:: make_unique<dropout>(p);
    }
    if(op=="linear"){
        int in,out;
        ss>>in>>out;
        return std:: make_unique<Linear>(in, out);
    }
    if(op=="conv2d"){
        int in, out, kernel, stride, padding;
        ss>>in>>out>>kernel>>stride>>padding;
        return std:: make_unique<conv2D>(in , out, kernel, stride, padding);
    }
    if(op=="maxpool"){

        int kernel; int stride;
        ss>>kernel>>stride;
        return std:: make_unique<maxpool2D>(kernel, stride);
    }
    if(op=="batchnorm1d"){
        int features;
        ss>>features;

        return std:: make_unique<batchnorm1D>(features);
    }
    if(op=="batchnorm2d"){
        int features;
        ss>>features;

        return std:: make_unique<batchnorm2D>(features);
    }
    if(op=="leakyrelu"){
        int alpha;
        ss>>alpha;
        return std:: make_unique<LeakyReLU>(alpha);
    }

    throw std::runtime_error("Unknown layer command: " + cmd);
}

