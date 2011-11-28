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
namespace accumulators { namespace server
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
    class managed_accumulator
      : public hpx::components::managed_component_base<managed_accumulator>
    {
    public:
        managed_accumulator() : value_(0) {}

        ///////////////////////////////////////////////////////////////////////
        // Exposed functionality of this component.

        /// Reset the value to 0.
        void reset()
        {
            value_.store(0);
        }

        /// Add the given number to the accumulator.
        void add(boost::uint64_t arg)
        {
            value_.fetch_add(arg);
        }

        /// Return the current value to the caller.
        boost::uint64_t query()
        {
            return value_.load();
        }

        ///////////////////////////////////////////////////////////////////////
        // Each of the exposed functions needs to be encapsulated into an
        // action type, generating all required boilerplate code for threads,
        // serialization, etc.

        /// Action codes. 
        enum actions
        {
            accumulator_reset = 0,
            accumulator_add   = 1,
            accumulator_query = 2
        };

        typedef hpx::actions::action0<
            // Component server type.
            managed_accumulator,
            // Action code.
            accumulator_reset,
            // Method bound to this action.
            &managed_accumulator::reset
        > reset_action;

        typedef hpx::actions::action1<
            // Component server type.
            managed_accumulator,
            // Action code.
            accumulator_add,
            // Arguments of this action.
            boost::uint64_t,
            // Method bound to this action.
            &managed_accumulator::add
        > add_action;

        typedef hpx::actions::result_action0<
            // Component server type.
            managed_accumulator,
            // Return type.
            boost::uint64_t,
            // Action code.
            accumulator_query,
            // Method bound to this action.
            &managed_accumulator::query
        > query_action;

    private:
        boost::atomic<boost::uint64_t> value_;
    };
}}

#endif

