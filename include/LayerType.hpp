#pragma once
#include <string>

enum class LayerType {
    ReLU,
    LeakyReLU,
    Sigmoid,
    Tanh,
    GELU,
    Softmax,
    Flatten,
    Dropout,
    Linear,
    Conv2D,
    MaxPool2D,
    BatchNorm1D,
    BatchNorm2D,
    Unknown
};

inline std::string layerTypeToString(LayerType type) {
    switch (type) {
        case LayerType::ReLU:        return "relu";
        case LayerType::LeakyReLU:   return "leakyrelu";
        case LayerType::Sigmoid:     return "sigmoid";
        case LayerType::Tanh:        return "tanh";
        case LayerType::GELU:        return "gelu";
        case LayerType::Softmax:     return "softmax";
        case LayerType::Flatten:     return "flatten";
        case LayerType::Dropout:     return "dropout";
        case LayerType::Linear:      return "linear";
        case LayerType::Conv2D:      return "conv2d";
        case LayerType::MaxPool2D:   return "maxpool2d";
        case LayerType::BatchNorm1D: return "batchnorm1d";
        case LayerType::BatchNorm2D: return "batchnorm2d";
        default:                     return "unknown";
    }
}
