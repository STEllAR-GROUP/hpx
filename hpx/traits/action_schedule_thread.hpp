//  Copyright (c) 2007-2016 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_TRAITS_ACTION_SCHEDULE_THREAD_MAR_30_2014_0325PM)
#define HPX_TRAITS_ACTION_SCHEDULE_THREAD_MAR_30_2014_0325PM

#include <hpx/runtime/naming/address.hpp>
#include <hpx/runtime/threads/thread_enums.hpp>
#include <hpx/runtime/threads/thread_init_data.hpp>
#include <hpx/traits/detail/wrap_int.hpp>

namespace hpx { namespace traits
{
    ///////////////////////////////////////////////////////////////////////////
    // Customization point for action capabilities
    namespace detail
    {
        struct schedule_thread_helper
        {
            // by default we return an empty value
            template <typename Action>
            static void
            call(wrap_int, naming::address::address_type,
                    naming::address::component_type,
                    threads::thread_init_data& data,
                    threads::thread_state_enum initial_state)
            {
                hpx::threads::register_work_plain(data, initial_state); //-V106
            }

            // forward the call if the component implements the function
            template <typename Action>
            static auto
            call(int, naming::address::address_type lva,
                    naming::address::component_type comptype,
                    threads::thread_init_data& data,
                    threads::thread_state_enum initial_state)
            ->  decltype(
                    Action::component_type::schedule_thread(
                        lva, comptype, data, initial_state)
                )
            {
                // by default we forward this to the component type
                typedef typename Action::component_type component_type;
                component_type::schedule_thread(lva, comptype, data, initial_state);
            }
        };

        template <typename Action>
        void call_schedule_thread(naming::address::address_type lva,
            naming::address::component_type comptype,
            threads::thread_init_data& data,
            threads::thread_state_enum initial_state)
        {
            schedule_thread_helper::template call<Action>(
                0, lva, comptype, data, initial_state);
        }
    }

    template <typename Action, typename Enable = void>
    struct action_schedule_thread
    {
        // returns whether target was migrated to another locality
        static void
        call(naming::address::address_type lva,
            naming::address::component_type comptype,
            threads::thread_init_data& data,
            threads::thread_state_enum initial_state)
        {
            return detail::call_schedule_thread<Action>(
                lva, comptype, data, initial_state);
        }
    };
}}

#endif

