//  Copyright (c) 2007-2008 Hartmut Kaiser
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_COMPONENTS_SERVER_ACCUMULATOR_MAY_17_2008_0731PM)
#define HPX_COMPONENTS_SERVER_ACCUMULATOR_MAY_17_2008_0731PM

#include <iostream>
#include <hpx/components/component_type.hpp>
#include <hpx/components/action.hpp>

namespace hpx { namespace components { namespace server
{
    ///////////////////////////////////////////////////////////////////////////
    /// The accumulator is a very simple example of a HPX component. 
    ///
    /// Note that the implementation of the accumulator does not require any 
    /// special data members or virtual functions.
    ///
    class accumulator 
    {
    public:
        // parcel action code: the action to be performed on the destination 
        // object (the accumulator)
        enum actions
        {
            init_accumulator = 0,
            add_to_accumulator = 1,
            print_accumulator = 2
        };

        // This is the component id. Every component needs to have an embedded
        // enumerator 'value' which is used by the generic action implementation
        // to associate this component with a given action.
        enum { value = component_accumulator };
        
        // constructor: initialize accumulator value
        accumulator()
          : arg_(0)
        {}
        
        ///////////////////////////////////////////////////////////////////////
        // exposed functionality of this component
        
        /// Initialize the accumulator
        bool init(hpx::threadmanager::px_thread_self&) 
        {
            arg_ = 0;
            return true;
        }
        
        /// Add the given number to the accumulator
        bool add (hpx::threadmanager::px_thread_self&, double arg) 
        {
            arg_ += arg;
            return true;
        }
        
        /// Print the current value of the accumulator
        bool print(hpx::threadmanager::px_thread_self&) 
        {
            std::cout << arg_ << std::endl;
            return true;
        }

        ///////////////////////////////////////////////////////////////////////
        // Each of the exposed functions needs to be encapsulated into a action
        // type, allowing to generate all require boilerplate code for threads,
        // serialization, etc.
        typedef action0<
            accumulator, init_accumulator, &accumulator::init
        > init_action;

        typedef action1<
            accumulator, add_to_accumulator, double, &accumulator::add
        > add_action;

        typedef action0<
            accumulator, print_accumulator, &accumulator::print
        > print_action;

    private:
        double arg_;
    };

}}}

#endif
