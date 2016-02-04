//  Copyright (c) 2007-2012 Hartmut Kaiser
//  Copyright (c)      2011 Bryce Adelstein-Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_EXAMPLES_SERVER_TEMPLATE_FUNCTION_ACCUMULATOR_JUL_12_2012_1056AM)
#define HPX_EXAMPLES_SERVER_TEMPLATE_FUNCTION_ACCUMULATOR_JUL_12_2012_1056AM

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
    ///     * Inherits from a component base class:
    ///       \a hpx::components::component_base
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
    class template_function_accumulator
      : public hpx::components::component_base<template_function_accumulator>
    {
    private:
        typedef hpx::lcos::local::spinlock mutex_type;

    public:
        template_function_accumulator() : value_(0) {}

        ///////////////////////////////////////////////////////////////////////
        // Exposed functionality of this component.

        /// Reset the value to 0.
        void reset()
        {
            // Atomically set value_ to 0.
            boost::lock_guard<mutex_type> l(mtx_);
            value_ = 0;
        }

        /// Add the given number to the accumulator.
        template <typename T>
        void add(T arg)
        {
            // Atomically add value_ to arg, and store the result in value_.
            boost::lock_guard<mutex_type> l(mtx_);
            value_ += boost::lexical_cast<double>(arg);
        }

        /// Return the current value to the caller.
        double query() const
        {
            // Get the value of value_.
            boost::lock_guard<mutex_type> l(mtx_);
            return value_;
        }

        ///////////////////////////////////////////////////////////////////////
        // Each of the exposed functions needs to be encapsulated into an
        // action type, generating all required boilerplate code for threads,
        // serialization, etc.

        HPX_DEFINE_COMPONENT_ACTION(template_function_accumulator, reset);
        HPX_DEFINE_COMPONENT_ACTION(template_function_accumulator, query);

        // Actions with template arguments (see add<>() above) require special
        // type definitions. The simplest way to define such an action type is
        // by deriving from the HPX facility make_action.
        template <typename T>
        struct add_action
          : hpx::actions::make_action<void (template_function_accumulator::*)(T),
                &template_function_accumulator::template add<T>, add_action<T> >
        {};

    private:
        mutable mutex_type mtx_;
        double value_;
    };
}}

HPX_REGISTER_ACTION_DECLARATION(
    examples::server::template_function_accumulator::reset_action,
    managed_accumulator_reset_action);

HPX_REGISTER_ACTION_DECLARATION(
    examples::server::template_function_accumulator::query_action,
    managed_accumulator_query_action);

#endif
