//  Copyright (c) 2007-2014 Hartmut Kaiser
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
    struct is_component;

    template <typename Component, typename Enable = void>
    struct component_type_database;

    template <typename Component, typename Enable = void>
    struct component_type_is_compatible;

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

    // Customization for action serialization filter
    template <typename Action, typename Enable = void>
    struct action_serialization_filter;

    template <typename Action, typename Enable = void>
    struct action_message_handler;

#if defined(HPX_HAVE_SECURITY)
    template <typename Action, typename Enable = void>
    struct action_capability_provider;
#endif

    template <typename Action, typename Enable = void>
    struct action_may_require_id_splitting;

    template <typename Action, typename Enable = void>
    struct action_is_target_valid;

    template <typename Action, typename Enable = void>
    struct action_does_termination_detection;

    template <typename Action, typename Enable = void>
    struct action_decorate_function;

    template <typename Action, typename Enable = void>
    struct action_decorate_continuation;

    template <typename Action, typename Enable = void>
    struct action_schedule_thread;

    ///////////////////////////////////////////////////////////////////////////
    // Customization point for type_size
    template <typename T, typename Enable = void>
    struct type_size;

    ///////////////////////////////////////////////////////////////////////////
    template <typename Future, typename Enable = void>
    struct is_future;

    template <typename Future, typename Enable = void>
    struct future_traits;

    ///////////////////////////////////////////////////////////////////////////
    template <typename Range, typename Enable = void>
    struct is_future_range;

    ///////////////////////////////////////////////////////////////////////////
    template <typename Tuple, typename Enable = void>
    struct is_future_tuple;

    ///////////////////////////////////////////////////////////////////////////
    template <typename Future, typename Enable = void>
    struct serialize_as_future;

    ///////////////////////////////////////////////////////////////////////////
    template <typename Plugin, typename Enable = void>
    struct component_config_data;

    ///////////////////////////////////////////////////////////////////////////
    template <typename Plugin, typename Enable = void>
    struct plugin_config_data;

    ///////////////////////////////////////////////////////////////////////////
    // Customization point for streaming with util::any
    template <typename T, typename Enable = void>
    struct supports_streaming_with_any;
}}

#endif
