//  Copyright (c) 2007-2020 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/coroutines/thread_enums.hpp>
#include <hpx/naming_base/naming_base.hpp>
#include <hpx/threading_base/register_thread.hpp>
#include <hpx/threading_base/thread_helpers.hpp>
#include <hpx/threading_base/thread_init_data.hpp>
#include <hpx/type_support/detail/wrap_int.hpp>

namespace hpx { namespace traits {

    ///////////////////////////////////////////////////////////////////////////
    // Customization point for action capabilities
    namespace detail {

        struct schedule_thread_helper
        {
            template <typename Action>
            static void call(wrap_int, naming::address_type,
                naming::component_type, threads::thread_init_data& data)
            {
                hpx::threads::register_work(data);
            }

            // forward the call if the component implements the function
            template <typename Action>
            static auto call(int, naming::address_type lva,
                naming::component_type comptype,
                threads::thread_init_data& data)
                -> decltype(Action::component_type::schedule_thread(
                    lva, comptype, data))
            {
                // by default we forward this to the component type
                using component_type = typename Action::component_type;
                component_type::schedule_thread(lva, comptype, data);
            }
        };
    }    // namespace detail

    template <typename Action, typename Enable = void>
    struct action_schedule_thread
    {
        static void call(naming::address_type lva,
            naming::component_type comptype, threads::thread_init_data& data)
        {
            return detail::schedule_thread_helper::template call<Action>(
                0, lva, comptype, data);
        }
    };
}}    // namespace hpx::traits
