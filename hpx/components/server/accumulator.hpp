//  Copyright (c) 2007-2008 Hartmut Kaiser
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_COMPONENTS_SERVER_ACCUMULATOR_MAY_17_2008_0731PM)
#define HPX_COMPONENTS_SERVER_ACCUMULATOR_MAY_17_2008_0731PM

#include <iostream>

#include <hpx/hpx_fwd.hpp>
#include <hpx/components/component_type.hpp>
#include <hpx/components/action.hpp>
#include <hpx/runtime/threadmanager/px_thread.hpp>

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
            query_accumulator_value = 2,
            print_accumulator = 3
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
        threadmanager::thread_state 
        init_proc (threadmanager::px_thread_self&, applier::applier& appl) 
        {
            arg_ = 0;
            return hpx::threadmanager::terminated;
        }

        /// Add the given number to the accumulator
        threadmanager::thread_state 
        add_proc (threadmanager::px_thread_self&, applier::applier& appl, double arg) 
        {
            arg_ += arg;
            return hpx::threadmanager::terminated;
        }

        /// Return the current value to the caller
        threadmanager::thread_state 
        query_proc (threadmanager::px_thread_self&, applier::applier& appl,
            double* result) 
        {
            // this will be zero if the action got invoked without continuations
            if (result)
                *result = arg_;
            return hpx::threadmanager::terminated;
        }

        /// Print the current value of the accumulator
        threadmanager::thread_state 
        print_proc (threadmanager::px_thread_self&, applier::applier& appl) 
        {
            std::cout << arg_ << std::endl;
            return hpx::threadmanager::terminated;
        }

        ///////////////////////////////////////////////////////////////////////
        // Each of the exposed functions needs to be encapsulated into an action
        // type, allowing to generate all required boilerplate code for threads,
        // serialization, etc.
        typedef action0<
            accumulator, init_accumulator, &accumulator::init_proc
        > init_action;

        typedef action1<
            accumulator, add_to_accumulator, double, &accumulator::add_proc
        > add_action;

        typedef result_action0<
            accumulator, double, query_accumulator_value, &accumulator::query_proc
        > query_action;

        typedef action0<
            accumulator, print_accumulator, &accumulator::print_proc
        > print_action;

    private:
        double arg_;
    };

}}}

#endif
