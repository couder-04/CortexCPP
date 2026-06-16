// If a function never touches member variables  make it static.

//  string -> object

// In a C++ class, the static keyword binds a member to the class itself rather than to individual objects

//  static binds the memeber to the class not tge object

# pragma once
# include <memory>
# include <string>
# include "Layer.hpp"

class LayerFactory{
    public:
        static std:: unique_ptr<Layer>create(const std:: string & cmd);
        // this function belongs to this class not a particular object  
};