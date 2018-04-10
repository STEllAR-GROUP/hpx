//  Copyright (c) 2018 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_TRAITS_ACTION_SELECT_DIRECT_EXECUTION_MAR_22_21018_0124PM)
#define HPX_TRAITS_ACTION_SELECT_DIRECT_EXECUTION_MAR_22_21018_0124PM

#include <hpx/config.hpp>
#include <hpx/runtime/launch_policy.hpp>
#include <hpx/runtime/naming_fwd.hpp>
#include <hpx/traits/detail/wrap_int.hpp>

namespace hpx { namespace traits
{
    ///////////////////////////////////////////////////////////////////////////
    // Customization point for action capabilities
    namespace detail
    {
        struct select_direct_execution_helper
        {
            // by default we return the unchanged function
            template <typename Action>
            static HPX_CONSTEXPR launch call(
                wrap_int, launch policy, naming::address_type)
            {
                return policy;
            }

            // forward the call if the component implements the function
            template <typename Action>
            static auto call(int, launch policy, naming::address_type lva)
            ->  decltype(
                    Action::component_type::select_direct_execution(
                        Action(), policy, lva)
                )
            {
                using component_type = typename Action::component_type;
                return component_type::select_direct_execution(
                    Action(), policy, lva);
            }
        };

        template <typename Action>
        HPX_CONSTEXPR launch call_select_direct_execution(
            launch policy, naming::address_type lva)
        {
            return select_direct_execution_helper::template call<Action>(
                0, policy, lva);
        }
    }

    template <typename Action, typename Enable = void>
    struct action_select_direct_execution
    {
        static HPX_CONSTEXPR launch call(launch policy,
            naming::address_type lva)
        {
            return detail::call_select_direct_execution<Action>(policy, lva);
        }
    };
}}

#endif

