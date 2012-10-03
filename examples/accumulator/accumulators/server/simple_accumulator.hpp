//  Copyright (c) 2007-2011 Hartmut Kaiser
//  Copyright (c)      2011 Bryce Adelstein-Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_4C46C86D_A43F_42A8_8164_C9EBA3B210CC)
#define HPX_4C46C86D_A43F_42A8_8164_C9EBA3B210CC

#include <hpx/hpx_fwd.hpp>
#include <hpx/runtime/components/server/simple_component_base.hpp>
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
    /// This example demonstrates how to write a simple component. Simple
    /// components are allocated one at a time with the C++'s new allocator.
    /// When a component needs to be created in small quantities, simple
    /// components should be used. At least two AGAS requests will be made when
    /// a simple component is created. 
    ///
    /// This component exposes 3 different actions: reset, add and query.  
    class simple_accumulator
      : public hpx::components::simple_component_base<simple_accumulator>
    {
    public:
        simple_accumulator() : value_(0) {}

        ///////////////////////////////////////////////////////////////////////
        // Exposed functionality of this component.

        /// Reset the components value to 0.
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
        boost::uint64_t query()
        {
            // Get the value of value_. 
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
            simple_accumulator,
            // Action code.
            accumulator_reset,
            // Method bound to this action.
            &simple_accumulator::reset
        > reset_action;

        typedef hpx::actions::action1<
            // Component server type.
            simple_accumulator,
            // Action code.
            accumulator_add,
            // Arguments of this action.
            boost::uint64_t,
            // Method bound to this action.
            &simple_accumulator::add
        > add_action;

        typedef hpx::actions::result_action0<
            // Component server type.
            simple_accumulator,
            // Return type.
            boost::uint64_t,
            // Action code.
            accumulator_query,
            // Method bound to this action.
            &simple_accumulator::query
        > query_action;

    private:
        boost::atomic<boost::uint64_t> value_;
    };
}}

HPX_REGISTER_ACTION_DECLARATION(
    examples::server::simple_accumulator::reset_action,
    simple_accumulator_reset_action);

HPX_REGISTER_ACTION_DECLARATION(
    examples::server::simple_accumulator::add_action,
    simple_accumulator_add_action);

HPX_REGISTER_ACTION_DECLARATION(
    examples::server::simple_accumulator::query_action,
    simple_accumulator_query_action);

#endif

