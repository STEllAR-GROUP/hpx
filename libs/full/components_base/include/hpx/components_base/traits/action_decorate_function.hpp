//  Copyright (c) 2007-2021 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/concepts/has_xxx.hpp>
#include <hpx/naming_base/naming_base.hpp>
#include <hpx/threading_base/threading_base_fwd.hpp>
#include <hpx/type_support/detail/wrap_int.hpp>

#include <utility>

namespace hpx { namespace traits {

    ///////////////////////////////////////////////////////////////////////////
    // Customization point for action capabilities
    namespace detail {

        // by default we return the unchanged function
        template <typename Component, typename F>
        F&& decorate_function(wrap_int, naming::address_type, F&& f) noexcept
        {
            return HPX_FORWARD(F, f);
        }

        // forward the call if the component implements the function
        template <typename Component, typename F>
        auto decorate_function(int, naming::address_type lva, F&& f)
            -> decltype(Component::decorate_action(lva, HPX_FORWARD(F, f)))
        {
            return Component::decorate_action(lva, HPX_FORWARD(F, f));
        }

        HPX_HAS_XXX_TRAIT_DEF(decorates_action)
    }    // namespace detail

    template <typename Action, typename Enable = void>
    struct has_decorates_action
      : detail::has_decorates_action<typename Action::component_type>
    {
    };

    template <typename Action>
    inline constexpr bool has_decorates_action_v =
        has_decorates_action<Action>::value;

    template <typename Action, typename Enable = void>
    struct action_decorate_function
    {
        static constexpr bool value = has_decorates_action_v<Action>;

        template <typename F>
        static threads::thread_function_type call(
            naming::address_type lva, F&& f)
        {
            using component_type = typename Action::component_type;
            return detail::decorate_function<component_type>(
                0, lva, HPX_FORWARD(F, f));
        }
    };

    template <typename Component, typename Enable = void>
    struct component_decorates_action : detail::has_decorates_action<Component>
    {
    };

    template <typename Component>
    inline constexpr bool component_decorates_action_v =
        component_decorates_action<Component>::value;

    template <typename Component, typename Enable = void>
    struct component_decorate_function
    {
        template <typename F>
        static threads::thread_function_type call(
            naming::address_type lva, F&& f)
        {
            return detail::decorate_function<Component>(
                0, lva, HPX_FORWARD(F, f));
        }
    };
}}    // namespace hpx::traits
