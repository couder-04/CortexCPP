#pragma once
#include "Tensor.hpp"
#include<string>

// A pure virtual function in C++ is a virtual function declared in a base class that forces all derived classes to provide their own implementation

class Layer{
    public:
//" virtual " means Child classes can Override this function i.e diff layers can have diff fwd passes
//  = 0 means that its an abstract class now : 
// An abstract class is a restricted blueprint that cannot be instantiated directly
// . It acts as a base class, NEEEDS to be inherited surely
        virtual Tensor forward (const Tensor &input)= 0;// fwd pass template

        virtual std:: vector<int> output_shape(const std::vector <int>&input_shape) const =0;// calculates the shape of output tensor

        virtual std:: string name() const =0;// architecture printing

        virtual long long  parameter_count() const =0; // parameter counting

        virtual ~ Layer() = default;// use default compiler generated destrcutor
        
// When deleting derived neural network layers through Layer pointers, destroy them correctly and safely.
};

// constructor: special function which starts automatically as the object is created
//      syntax:     Tensor(){
//                          function()      }
// never called , its just defined in public

// destructor: special function called when the Object dies: 
//  syntax:         ~Tensor();
//  defined in public... it cleans up the space]
// constructor called automatically->->object lives->scope ends->destructor called automatically

// virtual:
//          if there is some subclass which has a parent class
//          the parent class shares its members to the child classes
//          but if we want the child class to have some specialities in the same function
// then we use the virtual tag, it checks actual runtime object type

//  Pure virtual: (     "   =0  "      )
    //  "child classes MUST implement this"
    // makes the CLASS abstract . i.e it doesnt instantiate directly

//  virtual descructor:
//     virtual ~Layer()= default;
//  this calls the speific child class destructor first then the parent class destructor
//  without VIRTUAL we can only call Parent class destrcutor

// some behaviours of any Layer is same: like fwd pass etc hence we ,
// Have layer as the parent class and the Specific layer as the child class