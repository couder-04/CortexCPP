#include "LayerFactory.hpp"
#include <iostream>
#include <string>
#include <cassert>

int main() {
    std::cout << "--- Testing Base Layer Factory Commands ---\n";
    
    auto layer1 = LayerFactory::create("relu");
    std::cout << "relu name: " << layer1->name() << ", type enum: " << (int)layer1->type() << "\n";
    assert(layer1->type() == LayerType::ReLU);
    
    auto layer2 = LayerFactory::create("flatten");
    std::cout << "flatten name: " << layer2->name() << ", type enum: " << (int)layer2->type() << "\n";
    assert(layer2->type() == LayerType::Flatten);
    
    auto layer3 = LayerFactory::create("gelu");
    std::cout << "gelu name: " << layer3->name() << ", type enum: " << (int)layer3->type() << "\n";
    assert(layer3->type() == LayerType::GELU);
    
    auto layer4 = LayerFactory::create("sigmoid");
    std::cout << "sigmoid name: " << layer4->name() << ", type enum: " << (int)layer4->type() << "\n";
    assert(layer4->type() == LayerType::Sigmoid);
    
    auto layer5 = LayerFactory::create("dropout 0.3");
    std::cout << "dropout name: " << layer5->name() << ", type enum: " << (int)layer5->type() << "\n";
    assert(layer5->type() == LayerType::Dropout);
    
    auto layer6 = LayerFactory::create("softmax");
    std::cout << "softmax name: " << layer6->name() << ", type enum: " << (int)layer6->type() << "\n";
    assert(layer6->type() == LayerType::Softmax);
    
    auto layer7 = LayerFactory::create("linear 128 10");
    std::cout << "linear name: " << layer7->name() << ", type: " << (int)layer7->type() 
              << ", params: " << layer7->parameter_count() << "\n";
    assert(layer7->type() == LayerType::Linear);
    assert(layer7->parameter_count() == (128 * 10 + 10));
    
    auto layer8 = LayerFactory::create("conv2d 3 16 3 1 1");
    std::cout << "conv2d name: " << layer8->name() << ", type: " << (int)layer8->type() 
              << ", params: " << layer8->parameter_count() << "\n";
    assert(layer8->type() == LayerType::Conv2D);
    
    auto layer9 = LayerFactory::create("maxpool 2 2");
    std::cout << "maxpool name: " << layer9->name() << ", type: " << (int)layer9->type() << "\n";
    assert(layer9->type() == LayerType::MaxPool2D);
    
    auto layer10 = LayerFactory::create("batchnorm2d 16");
    std::cout << "batchnorm2d name: " << layer10->name() << ", type: " << (int)layer10->type() << "\n";
    assert(layer10->type() == LayerType::BatchNorm2D);
    
    auto layer11 = LayerFactory::create("batchnorm1d 16");
    std::cout << "batchnorm1d name: " << layer11->name() << ", type: " << (int)layer11->type() << "\n";
    assert(layer11->type() == LayerType::BatchNorm1D);
    
    auto layer12 = LayerFactory::create("leakyrelu 0.5");
    std::cout << "leakyrelu name: " << layer12->name() << ", type: " << (int)layer12->type() << "\n";
    assert(layer12->type() == LayerType::LeakyReLU);

    std::cout << "\n--- Testing Case-Insensitivity & LayerType creation ---\n";
    auto layer13 = LayerFactory::create("ReLU");
    std::cout << "ReLU (mixed-case) name: " << layer13->name() << "\n";
    assert(layer13->type() == LayerType::ReLU);
    
    auto layer14 = LayerFactory::create(LayerType::GELU);
    std::cout << "GELU (enum-created) name: " << layer14->name() << "\n";
    assert(layer14->type() == LayerType::GELU);

    std::cout << "\n--- Testing Dynamic Shape Propagation & Aliases ---\n";
    // Starting with input shape: {3, 32, 32}
    LayerFactory::set_input_shape({3, 32, 32});
    
    // conv 16 3 (should infer in_channels=3)
    auto conv_layer = LayerFactory::create("conv 16 3");
    std::cout << "conv 16 3 name: " << conv_layer->name() 
              << ", params: " << conv_layer->parameter_count() << "\n";
    // Check parameters: in_channel=3, out_channel=16, kernel=3. Stride=1, Padding=0.
    // Parameters: weights: 16*3*3*3 = 432, bias: 16. Total = 448 (Wait, parameter_count for Conv2D is in_channel*out_channel*kernel_size*kernel_size + out_channel)
    std::cout << "Expected params: " << (3 * 16 * 3 * 3 + 16) << ", Got: " << conv_layer->parameter_count() << "\n";
    assert(conv_layer->parameter_count() == (3 * 16 * 3 * 3 + 16));
    
    // Output shape should be {16, 30, 30} (kernel=3, stride=1, padding=0: 32 - 3 + 1 = 30)
    auto cur_shape = LayerFactory::get_current_shape();
    std::cout << "Current shape: (" << cur_shape[0] << "," << cur_shape[1] << "," << cur_shape[2] << ")\n";
    assert(cur_shape[0] == 16 && cur_shape[1] == 30 && cur_shape[2] == 30);
    
    // pool 2 (should infer stride = 2)
    auto pool_layer = LayerFactory::create("pool 2");
    std::cout << "pool 2 name: " << pool_layer->name() << "\n";
    
    // Output shape should be {16, 15, 15}
    cur_shape = LayerFactory::get_current_shape();
    std::cout << "Current shape after pool: (" << cur_shape[0] << "," << cur_shape[1] << "," << cur_shape[2] << ")\n";
    assert(cur_shape[0] == 16 && cur_shape[1] == 15 && cur_shape[2] == 15);
    
    // flatten
    auto flatten_layer = LayerFactory::create("flatten");
    cur_shape = LayerFactory::get_current_shape();
    std::cout << "Current shape after flatten: (" << cur_shape[0] << ")\n";
    assert(cur_shape.size() == 1 && cur_shape[0] == 3600);
    
    // linear 128 (should infer in_features=3600)
    auto linear_layer = LayerFactory::create("linear 128");
    std::cout << "linear 128 params: " << linear_layer->parameter_count() << "\n";
    assert(linear_layer->parameter_count() == (3600 * 128 + 128));

    std::cout << "\n--- Testing Error Handling ---\n";
    try {
        auto bad_layer = LayerFactory::create("apple");
        std::cout << "Error: created invalid layer apple!\n";
        return 1;
    } catch(const std::exception &e) {
        std::cout << "Caught expected exception for bad layer: " << e.what() << "\n";
    }

    std::cout << "\nALL TESTS PASSED SUCCESSFULLY!\n";
    return 0;
}