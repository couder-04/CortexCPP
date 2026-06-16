#include "ArchitectureAnalyzer.hpp"
#include "ArchitectureValidator.hpp"
#include "ParameterCounter.hpp"
#include <algorithm>    
#include <iostream>


static bool is_activation(const std::string& name) {
    return name == "relu"    || name == "leakyrelu" ||
           name == "sigmoid" || name == "tanh"      ||
           name == "gelu"    || name == "softmax";
}

int ArchitectureAnalyzer::score(sequential& model, std::vector<int>& input_shape) {

    int total = 100;

    
    if (!ArchitectureValidator::validate(model, input_shape))
        return 0;

    const auto& layers = model.get_layer();
    int n = static_cast<int>(layers.size());

    
    bool has_bn1d = false, has_bn2d = false;
    for (auto& l : layers) {
        if (l->name() == "batchnorm1d") has_bn1d = true;
        if (l->name() == "batchnorm2d") has_bn2d = true;
    }
    if (!has_bn1d && !has_bn2d) total -= 10;

    for (int i = 0; i < n; i++) {
        if (layers[i]->name() == "conv2d") {
            
            bool bn_nearby = (i+1 < n && layers[i+1]->name() == "batchnorm2d") ||
                             (i+2 < n && layers[i+2]->name() == "batchnorm2d");
            if (!bn_nearby) { total -= 10; break; }   
        }
    }

    
    bool has_linear = false, has_dropout = false;
    int  last_linear_idx = -1;                        
    for (int i = 0; i < n; i++) {
        if (layers[i]->name() == "linear")  { has_linear = true; last_linear_idx = i; }
        if (layers[i]->name() == "dropout")   has_dropout = true;
    }
    if (has_linear && !has_dropout) total -= 5;

    
    if (has_linear && last_linear_idx >= 0) {
        for (int i = last_linear_idx + 1; i < n; i++) {
            if (layers[i]->name() == "dropout") { total -= 5; break; }
        }
    }

    
    int act_count = 0;
    for (auto& l : layers)
        if (is_activation(l->name())) act_count++;

    if (act_count == 0) {
        total -= 15;
    } else {
        
        for (int i = 0; i < n; i++) {
            if (layers[i]->name() == "conv2d") {
                bool found = false;
                for (int j = i+1; j < n && j <= i+3; j++)
                    if (is_activation(layers[j]->name())) { found = true; break; }
                if (!found) { total -= 8; break; }
            }
        }
        
        for (int i = 0; i < n-1; i++) {
            if (layers[i]->name() == "sigmoid" || layers[i]->name() == "tanh") {
                total -= 5;
                break;
            }
        }
    }

    
    long long params = param_count::total_params(static_cast<const sequential&>(model));
    if      (params > 100'000'000) total -= 15;
    else if (params >  10'000'000) total -= 10;
    else if (params >   1'000'000) total -= 5;

    long long linear_params = 0;
    for (auto& l : layers)
        if (l->name() == "linear") linear_params += l->parameter_count();
    if (params > 0 && (linear_params * 100 / params) > 80) total -= 5;


    int struct_ded = 0;
    if (n < 3) struct_ded += 5;

    bool has_conv = false, has_maxpool = false, has_softmax = false;
    for (auto& l : layers) {
        if (l->name() == "conv2d")    has_conv    = true;
        if (l->name() == "maxpool2d") has_maxpool = true;  
        if (l->name() == "softmax")   has_softmax = true;
    }
    if (has_conv && !has_maxpool)                        struct_ded += 3;
    if (has_softmax && layers[n-1]->name() != "softmax") struct_ded += 5;

    int consec_lin = 0;
    for (auto& l : layers) {
        consec_lin = (l->name() == "linear") ? consec_lin + 1 : 0;
        if (consec_lin >= 3) { struct_ded += 5; break; }
    }
    total -= std::min(struct_ded, 15);

    return std::max(total, 1);   
}



std::vector<std::string> ArchitectureAnalyzer::suggestions(const sequential& model) {
    std::vector<std::string> out;
    const auto& layers = model.get_layer();
    int n = static_cast<int>(layers.size());

    bool has_bn1d = false, has_bn2d = false;
    bool has_conv = false, has_linear = false;
    bool has_dropout = false, has_maxpool = false;
    int  last_linear_idx = -1;

    for (int i = 0; i < n; i++) {
        const std::string& nm = layers[i]->name();
        if (nm == "conv2d")       has_conv    = true;
        if (nm == "linear")     { has_linear  = true; last_linear_idx = i; }
        if (nm == "dropout")      has_dropout = true;
        if (nm == "maxpool2d")    has_maxpool = true;
        if (nm == "batchnorm1d")  has_bn1d    = true;
        if (nm == "batchnorm2d")  has_bn2d    = true;
    }

    
    for (int i = 0; i < n; i++) {
        if (layers[i]->name() == "conv2d") {
            bool bn_nearby = (i+1 < n && layers[i+1]->name() == "batchnorm2d") ||
                             (i+2 < n && layers[i+2]->name() == "batchnorm2d");
            if (!bn_nearby) { out.push_back("Add BatchNorm2D after Conv2D"); break; }
        }
    }

    if (!has_bn1d && !has_bn2d)
        out.push_back("Add normalization layers to stabilize training");


    if (has_linear && !has_dropout)
        out.push_back("Consider Dropout before the final Linear layer");

    
    for (int i = 0; i < n; i++) {
        if (layers[i]->name() == "conv2d") {
            bool found = false;
            for (int j = i+1; j < n && j <= i+3; j++)
                if (is_activation(layers[j]->name())) { found = true; break; }
            if (!found) { out.push_back("Add an activation after Conv2D"); break; }
        }
    }

    
    for (int i = 0; i < n-1; i++) {
        if (layers[i]->name() == "sigmoid" || layers[i]->name() == "tanh") {
            out.push_back("Replace Sigmoid/Tanh in hidden layers with ReLU or GELU");
            break;
        }
    }

    
    if (has_conv && !has_maxpool)
        out.push_back("Add MaxPool2D after Conv blocks for spatial downsampling");

    
    if (has_linear && last_linear_idx >= 0) {
        for (int i = last_linear_idx + 1; i < n; i++) {
            if (layers[i]->name() == "dropout") {
                out.push_back("Move Dropout before the final Linear, not after");
                break;
            }
        }
    }

    return out;
}


std::vector<std::string> ArchitectureAnalyzer::warnings(const sequential& model) {
    std::vector<std::string> out;
    const auto& layers = model.get_layer();
    int n = static_cast<int>(layers.size());

    bool has_bn1d = false, has_bn2d = false;
    bool has_linear = false, has_dropout = false, has_softmax = false;

    for (int i = 0; i < n; i++) {
        const std::string& nm = layers[i]->name();
        if (nm == "linear")       has_linear  = true;
        if (nm == "dropout")      has_dropout = true;
        if (nm == "batchnorm1d")  has_bn1d    = true;
        if (nm == "batchnorm2d")  has_bn2d    = true;
        if (nm == "softmax")      has_softmax = true;
    }

    long long params = param_count::total_params(model);
    long long linear_params = 0;
    for (auto& l : layers)
        if (l->name() == "linear") linear_params += l->parameter_count();

    
    if      (params > 100'000'000) out.push_back("Very large model — consider reducing capacity");
    else if (params >  10'000'000) out.push_back("Large parameter count — risk of overfitting on small datasets");

    
    if (params > 0 && (linear_params * 100 / params) > 80)
        out.push_back("Large classifier head detected");

    
    if (!has_bn1d && !has_bn2d)
        out.push_back("No normalization layers found — training may be unstable");

    
    if (has_linear && !has_dropout)
        out.push_back("No Dropout found — model may overfit");

    
    bool saturating_hidden = false;
    for (int i = 0; i < n-1; i++)
        if (layers[i]->name() == "sigmoid" || layers[i]->name() == "tanh")
            saturating_hidden = true;
    if (saturating_hidden)
        out.push_back("Sigmoid or Tanh in hidden layers — vanishing gradient risk");


    if (has_softmax && layers[n-1]->name() != "softmax")
        out.push_back("Softmax is not the final layer — check output configuration");

    
    if (n < 3)
        out.push_back("Model is very shallow — consider adding more layers");

    return out;
}


void ArchitectureAnalyzer::analyze(sequential& model, std::vector<int>& input_shape) {

    int  sc    = score(model, input_shape);
    auto suggs = suggestions(static_cast<const sequential&>(model));
    auto warns = warnings(static_cast<const sequential&>(model));

    
    std::vector<std::string> bottlenecks;
    const auto& layers = model.get_layer();
    long long params = param_count::total_params(static_cast<const sequential&>(model));
    long long linear_params = 0;
    for (auto& l : layers)
        if (l->name() == "linear") linear_params += l->parameter_count();
    if (params > 0 && (linear_params * 100 / params) > 80)
        bottlenecks.push_back("Linear layer dominates parameter count");

    auto print_section = [](const std::vector<std::string>& v) {
        if (v.empty()) { std::cout << "  None\n"; return; }
        for (const auto& s : v) std::cout << "  - " << s << "\n";
    };

    std::cout << "\n=================================\n";
    std::cout <<   "ARCHITECTURE ANALYSIS\n";
    std::cout <<   "=================================\n\n";
    std::cout << "Architecture Score : " << sc << "/100\n";
    std::cout << "\nSuggestions:\n"; print_section(suggs);
    std::cout << "\nWarnings:\n";    print_section(warns);
    std::cout << "\nBottlenecks:\n"; print_section(bottlenecks);
    std::cout << "\n=================================\n";
}