//  Copyright (c) 2007-2015 Hartmut Kaiser
//  Copyright (c)      2011 Bryce Adelstein-Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_4C46C86D_A43F_42A8_8164_C9EBA3B210CC)
#define HPX_4C46C86D_A43F_42A8_8164_C9EBA3B210CC

#include <hpx/hpx_fwd.hpp>
#include <hpx/include/components.hpp>
#include <hpx/runtime/actions/component_action.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace examples { namespace server
{
    ///////////////////////////////////////////////////////////////////////////
    /// This class is a very simple example of an HPX component. An HPX
    /// component is a class that:
    ///
    ///     * Inherits from a component base class:
    ///       \a hpx::components::component_base
    ///     * Exposes methods that can be called asynchronously and/or remotely.
    ///       These constructs are known as HPX actions.
    ///
    /// By deriving this component from \a locking_hook the runtime system
    /// ensures that all action invocations are serialized. That means that
    /// the system ensures that no two actions are invoked at the same time on
    /// a given component instance. This makes the component thread safe and no
    /// additional locking has to be implemented by the user.
    ///
    /// Components are first-class objects in HPX. This means that they are
    /// globally addressable; all components have a unique GID.
    ///
    /// This example demonstrates how to write a simple component. Simple
    /// components are allocated one at a time with the C++'s new allocator.
    /// When a component needs to be created in small quantities, simple
    /// components should be used. At least two AGAS requests will be made when
    /// a simple component is created.
    ///
    /// This component exposes 3 different actions: reset, add and query.
    //[accumulator_server_inherit
    class accumulator
      : public hpx::components::locking_hook<
            hpx::components::component_base<accumulator> >
    //]
    {
    public:
        typedef boost::int64_t argument_type;

        //[accumulator_server_ctor
        accumulator() : value_(0) {}
        //]

        ///////////////////////////////////////////////////////////////////////
        // Exposed functionality of this component.

        //[accumulator_methods
        /// Reset the components value to 0.
        void reset()
        {
            //  set value_ to 0.
            value_ = 0;
        }

        /// Add the given number to the accumulator.
        void add(argument_type arg)
        {
            //  add value_ to arg, and store the result in value_.
            value_ += arg;
        }

        /// Return the current value to the caller.
        argument_type query() const
        {
            // Get the value of value_.
            return value_;
        }
        //]

        ///////////////////////////////////////////////////////////////////////
        // Each of the exposed functions needs to be encapsulated into an
        // action type, generating all required boilerplate code for threads,
        // serialization, etc.
        //[accumulator_action_types
        HPX_DEFINE_COMPONENT_ACTION(accumulator, reset);
        HPX_DEFINE_COMPONENT_ACTION(accumulator, add);
        HPX_DEFINE_COMPONENT_ACTION(accumulator, query);
        //]

    private:
        //[accumulator_server_data_member
        argument_type value_;
        //]
    };
}}

//[accumulator_registration_declarations
HPX_REGISTER_ACTION_DECLARATION(
    examples::server::accumulator::reset_action,
    accumulator_reset_action);

HPX_REGISTER_ACTION_DECLARATION(
    examples::server::accumulator::add_action,
    accumulator_add_action);

HPX_REGISTER_ACTION_DECLARATION(
    examples::server::accumulator::query_action,
    accumulator_query_action);
//]

#endif

