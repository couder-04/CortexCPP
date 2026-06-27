#include "LayerFactory.hpp"
#include "ReLU.hpp"
#include "LeakyReLU.hpp"
#include "Sigmoid.hpp"
#include "Tanh.hpp"
#include "GELU.hpp"
#include "Softmax.hpp"
#include "Flatten.hpp"
#include "Dropout.hpp"
#include "Linear.hpp"
#include "Conv2D.hpp"
#include "MaxPool2D.hpp"
#include "BatchNorm1D.hpp"
#include "BatchNorm2D.hpp"

#include <sstream>
#include <stdexcept>
#include <algorithm>
#include <cctype>
#include <numeric>

std::vector<int> LayerFactory::current_shape;

void LayerFactory::set_input_shape(const std::vector<int>& shape) {
    current_shape = shape;
}

const std::vector<int>& LayerFactory::get_current_shape() {
    return current_shape;
}

LayerType LayerFactory::string_to_type(const std::string & op) {
    std::string lower_op = op;
    std::transform(lower_op.begin(), lower_op.end(), lower_op.begin(), [](unsigned char c) {
        return std::tolower(c);
    });
    
    if (lower_op == "relu") return LayerType::ReLU;
    if (lower_op == "leakyrelu") return LayerType::LeakyReLU;
    if (lower_op == "sigmoid") return LayerType::Sigmoid;
    if (lower_op == "tanh") return LayerType::Tanh;
    if (lower_op == "gelu") return LayerType::GELU;
    if (lower_op == "softmax") return LayerType::Softmax;
    if (lower_op == "flatten") return LayerType::Flatten;
    if (lower_op == "dropout") return LayerType::Dropout;
    if (lower_op == "linear") return LayerType::Linear;
    if (lower_op == "conv2d" || lower_op == "conv") return LayerType::Conv2D;
    if (lower_op == "maxpool2d" || lower_op == "maxpool" || lower_op == "pool") return LayerType::MaxPool2D;
    if (lower_op == "batchnorm1d") return LayerType::BatchNorm1D;
    if (lower_op == "batchnorm2d") return LayerType::BatchNorm2D;
    return LayerType::Unknown;
}

std::unique_ptr<Layer> LayerFactory::create(LayerType type, const std::string & args) {
    std::string cmd = layerTypeToString(type);
    if (!args.empty()) {
        cmd += " " + args;
    }
    return create(cmd);
}

std::unique_ptr<Layer> LayerFactory::create(const std::string & cmd) {
    std::stringstream ss(cmd);
    std::string op;
    if (!(ss >> op)) {
        throw std::runtime_error("Empty layer command");
    }
    
    LayerType type = string_to_type(op);
    if (type == LayerType::Unknown) {
        throw std::runtime_error("Unknown layer command: " + cmd);
    }
    
    std::unique_ptr<Layer> layer;
    
    if (type == LayerType::ReLU) {
        layer = std::make_unique<ReLU>();
    }
    else if (type == LayerType::Flatten) {
        layer = std::make_unique<Flatten>();
    }
    else if (type == LayerType::Softmax) {
        layer = std::make_unique<softmax>();
    }
    else if (type == LayerType::GELU) {
        layer = std::make_unique<GELU>();
    }
    else if (type == LayerType::Sigmoid) {
        layer = std::make_unique<sigmoid>();
    }
    else if (type == LayerType::Tanh) {
        layer = std::make_unique<TanH>();
    }
    else if (type == LayerType::Dropout) {
        float p;
        if (!(ss >> p)) {
            throw std::runtime_error("Dropout requires probability value: " + cmd);
        }
        layer = std::make_unique<dropout>(p);
    }
    else if (type == LayerType::LeakyReLU) {
        float alpha = 0.01f;
        ss >> alpha; // Optional alpha
        layer = std::make_unique<LeakyReLU>(alpha);
    }
    else if (type == LayerType::Linear) {
        std::vector<int> params;
        int val;
        while (ss >> val) {
            params.push_back(val);
        }
        
        int in_features = 0;
        int out_features = 0;
        
        if (params.size() == 2) {
            in_features = params[0];
            out_features = params[1];
        } else if (params.size() == 1) {
            out_features = params[0];
            if (!current_shape.empty()) {
                long long product = 1;
                for (int dim : current_shape) {
                    product *= dim;
                }
                in_features = static_cast<int>(product);
            } else {
                in_features = out_features; // Default fallback if current_shape is not set
            }
        } else {
            throw std::runtime_error("Linear layer requires 1 or 2 integer parameters (out_features or in_features out_features): " + cmd);
        }
        
        layer = std::make_unique<Linear>(in_features, out_features);
    }
    else if (type == LayerType::Conv2D) {
        std::vector<int> params;
        int val;
        while (ss >> val) {
            params.push_back(val);
        }
        
        int in_channels = 0;
        int out_channels = 0;
        int kernel_size = 0;
        int stride = 1;
        int padding = 0;
        
        if (params.size() == 5) {
            in_channels = params[0];
            out_channels = params[1];
            kernel_size = params[2];
            stride = params[3];
            padding = params[4];
        } else if (params.size() == 3) {
            in_channels = params[0];
            out_channels = params[1];
            kernel_size = params[2];
        } else if (params.size() == 2) {
            out_channels = params[0];
            kernel_size = params[1];
            if (!current_shape.empty()) {
                in_channels = current_shape[0];
            } else {
                in_channels = out_channels; // Fallback
            }
        } else {
            throw std::runtime_error("Conv2D layer requires 2, 3, or 5 integer parameters: " + cmd);
        }
        
        layer = std::make_unique<conv2D>(in_channels, out_channels, kernel_size, stride, padding);
    }
    else if (type == LayerType::MaxPool2D) {
        std::vector<int> params;
        int val;
        while (ss >> val) {
            params.push_back(val);
        }
        
        int kernel_size = 0;
        int stride = 0;
        
        if (params.size() == 2) {
            kernel_size = params[0];
            stride = params[1];
        } else if (params.size() == 1) {
            kernel_size = params[0];
            stride = kernel_size; // default stride to kernel_size
        } else {
            throw std::runtime_error("MaxPool2D layer requires 1 or 2 integer parameters (kernel_size or kernel_size stride): " + cmd);
        }
        
        layer = std::make_unique<maxpool2D>(kernel_size, stride);
    }
    else if (type == LayerType::BatchNorm1D) {
        int features = 0;
        if (ss >> features) {
            // Explicit features
        } else {
            if (!current_shape.empty()) {
                features = current_shape[0];
            } else {
                throw std::runtime_error("BatchNorm1D requires features parameter or current input shape to be set: " + cmd);
            }
        }
        layer = std::make_unique<batchnorm1D>(features);
    }
    else if (type == LayerType::BatchNorm2D) {
        int features = 0;
        if (ss >> features) {
            // Explicit features
        } else {
            if (!current_shape.empty()) {
                features = current_shape[0];
            } else {
                throw std::runtime_error("BatchNorm2D requires features parameter or current input shape to be set: " + cmd);
            }
        }
        layer = std::make_unique<batchnorm2D>(features);
    }
    
    // Update internal shape state if valid input shape was set
    if (layer && !current_shape.empty()) {
        try {
            current_shape = layer->output_shape(current_shape);
        } catch (...) {
            // If shape propagation fails, clear/ignore
        }
    }
    
    return layer;
}
