// If a function never touches member variables  make it static.

//  string -> object

// In a C++ class, the static keyword binds a member to the class itself rather than to individual objects

//  static binds the memeber to the class not tge object

# pragma once
# include <memory>
# include <string>
# include <vector>
# include "Layer.hpp"
# include "LayerType.hpp"

class LayerFactory{
    private:
        static std::vector<int> current_shape;

    public:
        static void set_input_shape(const std::vector<int>& shape);
        static const std::vector<int>& get_current_shape();
        
        static std:: unique_ptr<Layer>create(const std:: string & cmd);
        static std:: unique_ptr<Layer>create(LayerType type, const std::string & args = "");
        
        static LayerType string_to_type(const std::string & op);
};