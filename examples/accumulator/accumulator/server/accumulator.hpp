//  Copyright (c) 2007-2012 Hartmut Kaiser
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
#include <hpx/runtime/actions/component_action.hpp>
#include <hpx/runtime/components/server/memory.hpp>

///////////////////////////////////////////////////////////////////////////////
//[accumulator_namespace
namespace hpx { namespace components { namespace server
{
    //
    ///////////////////////////////////////////////////////////////////////////
    /// The accumulator is a very simple example of a HPX component.
    ///
    /// The accumulator class is a small example component demonstrating the
    /// main principles of writing your own components. It exposes 4 different
    /// actions: init, add, query, and print, showing how to used and implement
    /// functionality in a way conformant with the HPX runtime system.
    ///
    /// Note that the implementation of the accumulator does not require any
    /// special data members or virtual functions.
    class accumulator
      : public components::detail::managed_component_base<accumulator>
    {
    public:
        // parcel action code: the action to be performed on the destination
        // object (the accumulator)

        //[accumulator_enum_action
        enum actions
        {
            accumulator_init = 0,
            accumulator_add = 1,
            accumulator_query_value = 2,
            accumulator_print = 3
        };
        //]

        // constructor: initialize accumulator value
        accumulator()
          : arg_(0)
        {}

        ///////////////////////////////////////////////////////////////////////
        // exposed functionality of this component

        //[accumulator_init
        /// Initialize the accumulator
        void init()
        {
            arg_ = 0;
        }
        //]

        /// Add the given number to the accumulator
        void add (unsigned long arg)
        {
            arg_ += arg;
        }

        /// Return the current value to the caller
        unsigned long query()
        {
            return arg_;
        }

        /// Print the current value of the accumulator
        void print()
        {
            applier::applier& appl = applier::get_applier();
            std::cout << appl.get_runtime_support_gid() << std::dec << "> "
                      << arg_ << std::flush << std::endl;
        }

        ///////////////////////////////////////////////////////////////////////
        // Each of the exposed functions needs to be encapsulated into an action
        // type, allowing to generate all required boilerplate code for threads,
        // serialization, etc.
        //[accumulator_action_init
        typedef hpx::actions::action0<
            accumulator, accumulator_init, &accumulator::init
        > init_action;
        //]

        //[accumulator_action_add
        typedef hpx::actions::action1<
            accumulator, accumulator_add, unsigned long, &accumulator::add
        > add_action;
        //]

        typedef hpx::actions::result_action0<
            accumulator, unsigned long, accumulator_query_value, &accumulator::query
        > query_action;

        typedef hpx::actions::action0<
            accumulator, accumulator_print, &accumulator::print
        > print_action;

    private:
        unsigned long arg_;
    };

}}}
//]


//[accumulator_action_declare

HPX_REGISTER_ACTION_DECLARATION_EX(
    hpx::components::server::accumulator::init_action,
    accumulator_init_action);
HPX_REGISTER_ACTION_DECLARATION_EX(
    hpx::components::server::accumulator::add_action,
    accumulator_add_action);
HPX_REGISTER_ACTION_DECLARATION_EX(
    hpx::components::server::accumulator::query_action,
    accumulator_query_action);
HPX_REGISTER_ACTION_DECLARATION_EX(
    hpx::components::server::accumulator::print_action,
    accumulator_print_action);
//]

#endif
