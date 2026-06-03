#pragma once
#include <vector> 
class sequential;   // initialized a class so that we can call its object later

namespace ArchitectureValidator{
    bool validate(sequential &model, std:: vector<int>&input_shape );
}