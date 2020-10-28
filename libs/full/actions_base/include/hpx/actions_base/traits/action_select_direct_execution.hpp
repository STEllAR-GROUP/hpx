//  Copyright (c) 2018-2020 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/async_base/launch_policy.hpp>
#include <hpx/naming_base/naming_base.hpp>
#include <hpx/type_support/detail/wrap_int.hpp>

namespace hpx { namespace traits {

    ///////////////////////////////////////////////////////////////////////////
    // Customization point for action capabilities
    namespace detail {

        struct select_direct_execution_helper
        {
            // by default we return the unchanged function
            template <typename Action>
            static constexpr launch call(
                wrap_int, launch policy, naming::address_type) noexcept
            {
                return policy;
            }

            // forward the call if the component implements the function
            template <typename Action>
            static auto call(int, launch policy, naming::address_type lva)
                -> decltype(Action::component_type::select_direct_execution(
                    Action(), policy, lva))
            {
                using component_type = typename Action::component_type;
                return component_type::select_direct_execution(
                    Action(), policy, lva);
            }
        };
    }    // namespace detail

    template <typename Action, typename Enable = void>
    struct action_select_direct_execution
    {
        static constexpr launch call(launch policy, naming::address_type lva)
        {
            return detail::select_direct_execution_helper::template call<
                Action>(0, policy, lva);
        }
    };
}}    // namespace hpx::traits
