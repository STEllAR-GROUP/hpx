//  Copyright (c) 2007-2012 Hartmut Kaiser
//  Copyright (c)      2011 Bryce Adelstein-Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_EXAMPLES_SERVER_TEMPLATE_ACCUMULATOR_JUL_11_2012_1239PM)
#define HPX_EXAMPLES_SERVER_TEMPLATE_ACCUMULATOR_JUL_11_2012_1239PM

#include <hpx/hpx_fwd.hpp>
#include <hpx/include/components.hpp>
#include <hpx/include/local_lcos.hpp>

#include <boost/lexical_cast.hpp>

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
    class template_accumulator
      : public hpx::components::managed_component_base<template_accumulator>
    {
    private:
        typedef hpx::lcos::local::spinlock mutex_type;

    public:
        template_accumulator() : value_(0) {}

        ///////////////////////////////////////////////////////////////////////
        // Exposed functionality of this component.

        /// Reset the value to 0.
        void reset()
        {
            // Atomically set value_ to 0.
            mutex_type::scoped_lock l(mtx_);
            value_ = 0;
        }

        /// Add the given number to the accumulator.
        template <typename T>
        void add(T arg)
        {
            // Atomically add value_ to arg, and store the result in value_.
            mutex_type::scoped_lock l(mtx_);
            value_ += boost::lexical_cast<double>(arg);
        }

        /// Return the current value to the caller.
        double query() const
        {
            // Get the value of value_.
            mutex_type::scoped_lock l(mtx_);
            return value_;
        }

        ///////////////////////////////////////////////////////////////////////
        // Each of the exposed functions needs to be encapsulated into an
        // action type, generating all required boilerplate code for threads,
        // serialization, etc.

        HPX_DEFINE_COMPONENT_ACTION(template_accumulator, reset, reset_action);
        HPX_DEFINE_COMPONENT_CONST_ACTION(template_accumulator, query, query_action);

        // Actions with template arguments (see add<>() above) require special
        // type definitions. The simplest way to define such an action type is
        // by deriving from the HPX facility make_action:
        template <typename T>
        struct add_action
          : hpx::actions::make_action<
                void (template_accumulator::*)(T),
                &template_accumulator::template add<T>,
                boost::mpl::false_, add_action<T> >
        {};

    private:
        mutable mutex_type mtx_;
        double value_;
    };
}}

HPX_REGISTER_ACTION_DECLARATION_EX(
    examples::server::template_accumulator::reset_action,
    managed_accumulator_reset_action);

HPX_REGISTER_ACTION_DECLARATION_EX(
    examples::server::template_accumulator::query_action,
    managed_accumulator_query_action);

// Actions with template arguments do not need to be declared in the same way
// as all other actions. The reason is that it is impossible to declare the
// action with all possible template argument type combinations. Thus, we let
// the compiler generate the necessary serialization code. The same technique
// could be employed for non-template action-types as well, however this
// increases compilation time considerably.
HPX_SERIALIZATION_REGISTER_TEMPLATE_ACTION(
    (template <typename T>),
    (examples::server::template_accumulator::add_action<T>)
)

#endif

