//  Copyright (c) 2007-2014 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_TRAITS_ACTION_SCHEDULE_THREAD_MAR_30_2014_0325PM)
#define HPX_TRAITS_ACTION_SCHEDULE_THREAD_MAR_30_2014_0325PM

#include <hpx/runtime/naming/address.hpp>
#include <hpx/runtime/threads/thread_enums.hpp>
#include <hpx/runtime/threads/thread_init_data.hpp>
#include <hpx/util/always_void.hpp>

namespace hpx { namespace traits
{
    ///////////////////////////////////////////////////////////////////////////
    // Customization point for action capabilities
    template <typename Action, typename Enable>
    struct action_schedule_thread
    {
        static void
        call(naming::address::address_type lva, threads::thread_init_data& data,
            threads::thread_state_enum initial_state)
        {
            // by default we forward this to the component type
            typedef typename Action::component_type component_type;
            return component_type::schedule_thread(lva, data, initial_state);
        }
    };

    template <typename Action>
    struct action_schedule_thread<Action
      , typename util::always_void<typename Action::type>::type>
      : action_schedule_thread<typename Action::type>
    {};
}}

#endif

