//  Copyright (c) 2007-2012 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_TRAITS_OCT_26_2011_0838AM)
#define HPX_TRAITS_OCT_26_2011_0838AM

namespace hpx { namespace traits
{
    ///////////////////////////////////////////////////////////////////////////
    template <typename Result, typename Enable = void>
    struct promise_remote_result;

    template <typename Result, typename Enable = void>
    struct promise_local_result;

    ///////////////////////////////////////////////////////////////////////////
    template <typename Result, typename RemoteResult, typename Enable = void>
    struct get_remote_result;

    ///////////////////////////////////////////////////////////////////////////
    template <typename Component, typename Enable = void>
    struct component_type_database;

    ///////////////////////////////////////////////////////////////////////////
    // control the way managed_components are constructed
    struct construct_with_back_ptr {};
    struct construct_without_back_ptr {};

    template <typename T, typename Enable = void>
    struct managed_component_ctor_policy
    {
        typedef construct_without_back_ptr type;
    };

    ///////////////////////////////////////////////////////////////////////////
    // control the way managed_components are destructed
    struct managed_object_is_lifetime_controlled {};
    struct managed_object_controls_lifetime {};

    template <typename T, typename Enable = void>
    struct managed_component_dtor_policy
    {
        typedef managed_object_controls_lifetime type;
    };

    ///////////////////////////////////////////////////////////////////////////
    template <typename Action, typename Enable = void>
    struct needs_guid_initialization;

    ///////////////////////////////////////////////////////////////////////////
    template <typename Action, typename Enable = void>
    struct is_action;

    // Customization point for action priority
    template <typename Action, typename Enable = void>
    struct action_priority;

    // Customization point for action stack size
    template <typename Action, typename Enable = void>
    struct action_stacksize;

    ///////////////////////////////////////////////////////////////////////////
    // Customization point for argument_size
    template <typename T, typename Enable = void>
    struct argument_size;

    ///////////////////////////////////////////////////////////////////////////
    template <typename Future, typename Enable = void>
    struct is_future;

    ///////////////////////////////////////////////////////////////////////////
    template <typename Component, typename Enable = void>
    struct is_component;
}}

#endif
