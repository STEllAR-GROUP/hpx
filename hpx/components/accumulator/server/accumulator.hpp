//  Copyright (c) 2007-2008 Hartmut Kaiser
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_COMPONENTS_SERVER_ACCUMULATOR_MAY_17_2008_0731PM)
#define HPX_COMPONENTS_SERVER_ACCUMULATOR_MAY_17_2008_0731PM

#include <iostream>

#include <hpx/hpx_fwd.hpp>
#include <hpx/runtime/applier/applier.hpp>
#include <hpx/runtime/threads/thread.hpp>
#include <hpx/runtime/components/component_type.hpp>
#include <hpx/runtime/components/server/managed_component_base.hpp>
#include <hpx/runtime/actions/action.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace components { namespace server 
{
    // forward declaration
    class accumulator;

}}}

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace components { namespace server { namespace detail
{
    ///////////////////////////////////////////////////////////////////////////
    /// The accumulator is a very simple example of a HPX component. 
    ///
    /// Note that the implementation of the accumulator does not require any 
    /// special data members or virtual functions.
    ///
    class accumulator 
    {
    private:
        static component_type value;

    public:
        // components must contain a typedef for wrapping_type defining the
        // managed_component_base type used to encapsulate instances of this 
        // component
        typedef 
            managed_component_base<accumulator, server::accumulator> 
        wrapping_type;

        // parcel action code: the action to be performed on the destination 
        // object (the accumulator)
        enum actions
        {
            accumulator_init = 0,
            accumulator_add = 1,
            accumulator_query_value = 2,
            accumulator_print = 3
        };

        // This is the component id. Every component needs to have an embedded
        // enumerator 'value' which is used by the generic action implementation
        // to associate this component with a given action.
        static HPX_COMPONENT_EXPORT component_type get_component_type();
        static void set_component_type(component_type);

        // constructor: initialize accumulator value
        accumulator()
          : arg_(0)
        {}

        ///////////////////////////////////////////////////////////////////////
        // exposed functionality of this component

        /// Initialize the accumulator
        threads::thread_state 
        init (threads::thread_self&, applier::applier& appl) 
        {
            arg_ = 0;
            return threads::terminated;
        }

        /// Add the given number to the accumulator
        threads::thread_state 
        add (threads::thread_self&, applier::applier& appl, double arg) 
        {
            arg_ += arg;
            return threads::terminated;
        }

        /// Return the current value to the caller
        threads::thread_state 
        query (threads::thread_self&, applier::applier& appl,
            double* result) 
        {
            // this will be zero if the action got invoked without continuations
            if (result)
                *result = arg_;
            return threads::terminated;
        }

        /// Print the current value of the accumulator
        threads::thread_state 
        print (threads::thread_self&, applier::applier& appl) 
        {
            std::cout << arg_ << std::flush << std::endl;
            return threads::terminated;
        }

        ///////////////////////////////////////////////////////////////////////
        // Each of the exposed functions needs to be encapsulated into an action
        // type, allowing to generate all required boilerplate code for threads,
        // serialization, etc.
        typedef hpx::actions::action0<
            accumulator, accumulator_init, &accumulator::init
        > init_action;

        typedef hpx::actions::action1<
            accumulator, accumulator_add, double, &accumulator::add
        > add_action;

        typedef hpx::actions::result_action0<
            accumulator, double, accumulator_query_value, &accumulator::query
        > query_action;

        typedef hpx::actions::action0<
            accumulator, accumulator_print, &accumulator::print
        > print_action;

    private:
        double arg_;
    };

}}}}

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace components { namespace server 
{
    /// \class accumulator accumulator.hpp hpx/components/accumulator.hpp
    ///
    /// The accumulator class is a small example components demonstrating the
    /// main principles of writing your own components. It exposes 4 different
    /// actions: init, add, query, and print, showing how to used and implement
    /// functionality in a way conformant with the HPX runtime system. 
    class accumulator 
      : public managed_component_base<detail::accumulator, accumulator>
    {
    private:
        typedef detail::accumulator wrapped_type;
        typedef managed_component_base<wrapped_type, accumulator> base_type;

    public:
        accumulator(applier::applier&)
          : base_type(new wrapped_type())
        {}
    };

}}}

#endif
