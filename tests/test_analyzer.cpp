#include "ArchitectureAnalyzer.hpp"
#include "GELU.hpp"
#include "Softmax.hpp"
#include "Sigmoid.hpp"
#include "Sequential.hpp"
#include "Conv2D.hpp"
#include "BatchNorm2D.hpp"
#include "ReLU.hpp"
#include "MaxPool2D.hpp"
#include "Flatten.hpp"
#include "Linear.hpp"
#include"Dropout.hpp"

int main(){


std::vector<int> input_shape = {3,32,32};


sequential model;

model.add(std::make_unique<conv2D>(3,16,3,1,1));
model.add(std::make_unique<batchnorm2D>(16));
model.add(std::make_unique<ReLU>());

model.add(std::make_unique<maxpool2D>(2,2));

model.add(std::make_unique<conv2D>(16,32,3,1,1));
model.add(std::make_unique<batchnorm2D>(32));
model.add(std::make_unique<ReLU>());

model.add(std::make_unique<maxpool2D>(2,2));

model.add(std::make_unique<Flatten>());
model.add(std::make_unique<dropout>(0.5));

model.add(std::make_unique<Linear>(2048,10));
model.add(std::make_unique<sigmoid>());

for(const auto& layer : model.get_layer()){
    std::cout << "[" << layer->name() << "]\n";
}
ArchitectureAnalyzer::analyze(model, input_shape);


sequential bad_cnn;

bad_cnn.add(std::make_unique<conv2D>(3,16,3,1,1));
bad_cnn.add(std::make_unique<Flatten>());
bad_cnn.add(std::make_unique<Linear>(16384,10));
for(const auto& layer : bad_cnn.get_layer()){
    std::cout << "[" << layer->name() << "]\n";
}
ArchitectureAnalyzer::analyze(bad_cnn, input_shape);


sequential faulty;

faulty.add(std::make_unique<conv2D>(3,16,3,1,1));
faulty.add(std::make_unique<Linear>(100,10));
for(const auto& layer : bad_cnn.get_layer()){
    std::cout << "[" << layer->name() << "]\n";
}
ArchitectureAnalyzer::analyze(faulty, input_shape);

sequential mlp;

mlp.add(std::make_unique<Linear>(784,512));
mlp.add(std::make_unique<GELU>());

mlp.add(std::make_unique<dropout>(0.3));

mlp.add(std::make_unique<Linear>(512,256));
mlp.add(std::make_unique<GELU>());

mlp.add(std::make_unique<dropout>(0.3));

mlp.add(std::make_unique<Linear>(256,10));

mlp.add(std::make_unique<softmax>());

    return 0;
}