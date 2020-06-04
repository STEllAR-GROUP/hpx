//  Copyright (c) 2007-2016 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/runtime/naming/address.hpp>
#include <hpx/coroutines/thread_enums.hpp>
#include <hpx/modules/naming_base.hpp>
#include <hpx/threading_base/thread_helpers.hpp>
#include <hpx/threading_base/thread_init_data.hpp>
#include <hpx/type_support/detail/wrap_int.hpp>

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
                    threads::thread_init_data& data)
            {
                hpx::threads::register_work(data); //-V106
            }

            // forward the call if the component implements the function
            template <typename Action>
            static auto
            call(int, naming::address::address_type lva,
                    naming::address::component_type comptype,
                    threads::thread_init_data& data)
            ->  decltype(
                    Action::component_type::schedule_thread(
                        lva, comptype, data)
                )
            {
                // by default we forward this to the component type
                typedef typename Action::component_type component_type;
                component_type::schedule_thread(lva, comptype, data);
            }
        };

        template <typename Action>
        void call_schedule_thread(naming::address::address_type lva,
            naming::address::component_type comptype,
            threads::thread_init_data& data)
        {
            schedule_thread_helper::template call<Action>(
                0, lva, comptype, data);
        }
    }

    template <typename Action, typename Enable = void>
    struct action_schedule_thread
    {
        // returns whether target was migrated to another locality
        static void
        call(naming::address::address_type lva,
            naming::address::component_type comptype,
            threads::thread_init_data& data)
        {
            return detail::call_schedule_thread<Action>(
                lva, comptype, data);
        }
    };
}}


