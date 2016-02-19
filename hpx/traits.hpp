//  Copyright (c) 2007-2016 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_TRAITS_OCT_26_2011_0838AM)
#define HPX_TRAITS_OCT_26_2011_0838AM

#include <hpx/config.hpp>

namespace hpx { namespace traits
{
    namespace detail
    {
        // wraps int so that int argument is favored over wrap_int
        struct wrap_int
        {
            HPX_CONSTEXPR wrap_int(int) {}
        };
    }

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
    struct is_component_or_component_array;

    template <typename Component, typename Enable = void>
    struct component_type_database;

    template <typename Component, typename Enable = void>
    struct component_type_is_compatible;

    template <typename Component, typename Enable = void>
    struct component_supports_migration;

    ///////////////////////////////////////////////////////////////////////////
    template <typename T, typename Enable = void>
    struct is_client;

    template <typename T, typename Enable = void>
    struct is_client_or_client_array;

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
    struct is_action;

    template <typename Action, typename Enable = void>
    struct is_continuation;

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
    struct action_is_target_valid;

    template <typename Action, typename Enable = void>
    struct action_does_termination_detection;

    template <typename Action, typename Enable = void>
    struct action_decorate_function;

    template <typename Action, typename Enable = void>
    struct action_decorate_continuation;

    template <typename Action, typename Enable = void>
    struct action_schedule_thread;

    template <typename Action, typename Enable = void>
    struct action_was_object_migrated;

    ///////////////////////////////////////////////////////////////////////////
    template <typename A, typename Enable = void>
    struct is_chunk_allocator;

    template <typename A, typename Enable = void>
    struct default_chunk_size;

    ///////////////////////////////////////////////////////////////////////////
    template <typename Range, typename Enable = void>
    struct is_range;

    template <typename Future, typename Enable = void>
    struct is_future;

    template <typename Future, typename Enable = void>
    struct future_traits;

    template <typename Future, typename Enable = void>
    struct future_access;

    template <typename Range, typename Enable = void>
    struct is_future_range;

    template <typename Range, typename Enable = void>
    struct future_range_traits;

    template <typename Tuple, typename Enable = void>
    struct is_future_tuple;

    template <typename Future, typename Enable = void>
    struct acquire_future;

    template <typename Future, typename Enable = void>
    struct acquire_shared_state;

    template <typename T, typename Enable = void>
    struct is_shared_state;

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

    ///////////////////////////////////////////////////////////////////////////
    template <typename Iterator, typename Enable = void>
    struct segmented_iterator_traits;

    template <typename Iterator, typename Enable = void>
    struct is_segmented_iterator;

    template <typename Iterator, typename Enable = void>
    struct segmented_local_iterator_traits;

    template <typename T, typename Enable = void>
    struct projected_iterator;

    ///////////////////////////////////////////////////////////////////////////
    template <typename T, typename Enable = void>
    struct is_distribution_policy;

    template <typename T, typename Enable = void>
    struct is_executor;

    template <typename T, typename Enable = void>
    struct is_timed_executor;

    template <typename T, typename Enable = void>
    struct is_executor_parameters;
}}

#endif
