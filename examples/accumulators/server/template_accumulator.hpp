//  Copyright (c) 2007-2016 Hartmut Kaiser
//  Copyright (c)      2011 Bryce Adelstein-Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_TEMPLATE_ACCUMULATOR_SERVER_MAR_31_2016_1040AM)
#define HPX_TEMPLATE_ACCUMULATOR_SERVER_MAR_31_2016_1040AM

#include <hpx/include/components.hpp>
#include <hpx/include/actions.hpp>

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
    template <typename T>
    class template_accumulator
      : public hpx::components::locking_hook<
            hpx::components::component_base<template_accumulator<T> > >
    {
    public:
        typedef T argument_type;

        template_accumulator() : value_(0) {}

        ///////////////////////////////////////////////////////////////////////
        // Exposed functionality of this component.

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

        ///////////////////////////////////////////////////////////////////////
        // Each of the exposed functions needs to be encapsulated into an
        // action type, generating all required boilerplate code for threads,
        // serialization, etc.
        HPX_DEFINE_COMPONENT_ACTION(template_accumulator, reset);
        HPX_DEFINE_COMPONENT_ACTION(template_accumulator, add);
        HPX_DEFINE_COMPONENT_ACTION(template_accumulator, query);

    private:
        argument_type value_;
    };
}}

#define REGISTER_TEMPLATE_ACCUMULATOR_DECLARATION(type)                       \
    HPX_REGISTER_ACTION_DECLARATION(                                          \
        examples::server::template_accumulator<type>::reset_action,           \
        HPX_PP_CAT(__template_accumulator_reset_action_, type));              \
                                                                              \
    HPX_REGISTER_ACTION_DECLARATION(                                          \
        examples::server::template_accumulator<type>::add_action,             \
        HPX_PP_CAT(__template_accumulator_add_action_, type));                \
                                                                              \
    HPX_REGISTER_ACTION_DECLARATION(                                          \
        examples::server::template_accumulator<type>::query_action,           \
        HPX_PP_CAT(__template_accumulator_query_action_, type));              \
/**/

#define REGISTER_TEMPLATE_ACCUMULATOR(type)                                   \
    HPX_REGISTER_ACTION(                                                      \
        examples::server::template_accumulator<type>::reset_action,           \
        HPX_PP_CAT(__template_accumulator_reset_action_, type));              \
                                                                              \
    HPX_REGISTER_ACTION(                                                      \
        examples::server::template_accumulator<type>::add_action,             \
        HPX_PP_CAT(__template_accumulator_add_action_, type));                \
                                                                              \
    HPX_REGISTER_ACTION(                                                      \
        examples::server::template_accumulator<type>::query_action,           \
        HPX_PP_CAT(__template_accumulator_query_action_, type));              \
                                                                              \
    typedef ::hpx::components::component<                                     \
        examples::server::template_accumulator<type>                          \
    > HPX_PP_CAT(__template_accumulator_, type);                              \
    HPX_REGISTER_COMPONENT(HPX_PP_CAT(__template_accumulator_, type))         \
/**/

#endif

