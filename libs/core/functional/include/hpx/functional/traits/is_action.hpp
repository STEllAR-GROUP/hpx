//  Copyright (c) 2007-2022 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>

#include <type_traits>

namespace hpx::traits {

    namespace detail {

        template <typename Action, typename Enable = void>
        struct is_action_impl : std::false_type
        {
        };

        template <typename Action>
        struct is_action_impl<Action, std::void_t<typename Action::action_tag>>
          : std::true_type
        {
        };
    }    // namespace detail

    template <typename Action, typename Enable = void>
    struct is_action : detail::is_action_impl<std::decay_t<Action>>
    {
    };

    template <typename T>
    inline constexpr bool is_action_v = is_action<T>::value;
}    // namespace hpx::traits

namespace hpx {

    ///////////////////////////////////////////////////////////////////////////
    template <typename Action>
    struct is_bound_action : std::false_type
    {
    };

    template <typename T>
    inline constexpr bool is_bound_action_v = is_bound_action<T>::value;
}    // namespace hpx

namespace hpx::traits {

    template <typename Action>
    using is_bound_action HPX_DEPRECATED_V(1, 8,
        "hpx::traits::is_bound_action is deprecated, use "
        "hpx::is_bound_action instead") = hpx::is_bound_action<Action>;

    template <typename Action>
    HPX_DEPRECATED_V(1, 8,
        "hpx::traits::is_bound_action_v is deprecated, use "
        "hpx::is_bound_action_v instead")
    inline constexpr bool is_bound_action_v = hpx::is_bound_action_v<Action>;
}    // namespace hpx::traits
