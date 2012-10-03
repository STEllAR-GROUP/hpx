//  Copyright (c) 2007-2011 Hartmut Kaiser
//  Copyright (c)      2011 Bryce Adelstein-Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_B808C8CA_810E_4583_9EA2_528553C8B511)
#define HPX_B808C8CA_810E_4583_9EA2_528553C8B511

#include <hpx/hpx_fwd.hpp>
#include <hpx/runtime/components/server/managed_component_base.hpp>
#include <hpx/runtime/actions/component_action.hpp>

#include <boost/atomic.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace examples { namespace server
{
    ///////////////////////////////////////////////////////////////////////////
    /// This class is a very simple example of an HPX component. An HPX
    /// component is a class that:
    ///
    ///     * Inherits from a component base class (either
    ///       \a hpx::components::managed_component_base or
    ///       \a hpx::components::simple_component_base).
    ///     * Exposes methods that can be called asynchronously and/or remotely.
    ///       These constructs are known as HPX actions.
    ///
    /// Components are first-class objects in HPX. This means that they are
    /// globally addressable; all components have a unique GID.
    ///
    /// This example demonstrates how to write a managed component. Managed
    /// components are allocated in bulk by HPX. When a component needs to be
    /// created in large quantities, managed components should be used. Because
    /// managed components are allocated in bulk, the creation of a new managed
    /// component usually does not require AGAS requests.
    ///
    /// This component exposes 3 different actions: reset, add and query.
    //[managed_accumulator_server_inherit
    class managed_accumulator
      : public hpx::components::managed_component_base<managed_accumulator>
    //]
    {
    public:
        //[managed_accumulator_server_ctor
        managed_accumulator() : value_(0) {}
        //]

        ///////////////////////////////////////////////////////////////////////
        // Exposed functionality of this component.

        //[managed_accumulator_methods
        /// Reset the value to 0.
        void reset()
        {
            // Atomically set value_ to 0.
            value_.store(0);
        }

        /// Add the given number to the accumulator.
        void add(boost::uint64_t arg)
        {
            // Atomically add value_ to arg, and store the result in value_.
            value_.fetch_add(arg);
        }

        /// Return the current value to the caller.
        boost::uint64_t query() const
        {
            // Get the value of value_.
            return value_.load();
        }
        //]

        ///////////////////////////////////////////////////////////////////////
        // Each of the exposed functions needs to be encapsulated into an
        // action type, generating all required boilerplate code for threads,
        // serialization, etc.

        //[managed_accumulator_action_types
        HPX_DEFINE_COMPONENT_ACTION(managed_accumulator, reset, reset_action);
        HPX_DEFINE_COMPONENT_ACTION(managed_accumulator, add, add_action);
        HPX_DEFINE_COMPONENT_CONST_ACTION(managed_accumulator, query, query_action);
        //]

    //[managed_accumulator_server_data_member
    private:
        boost::atomic<boost::uint64_t> value_;
    //]
    };
}}

//[managed_accumulator_registration_declarations
HPX_REGISTER_ACTION_DECLARATION(
    examples::server::managed_accumulator::reset_action,
    managed_accumulator_reset_action);

HPX_REGISTER_ACTION_DECLARATION(
    examples::server::managed_accumulator::add_action,
    managed_accumulator_add_action);

HPX_REGISTER_ACTION_DECLARATION(
    examples::server::managed_accumulator::query_action,
    managed_accumulator_query_action);
//]

#endif

