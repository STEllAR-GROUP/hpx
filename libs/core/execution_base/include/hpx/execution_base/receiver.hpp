//  Copyright (c) 2020 Thomas Heller
//  Copyright (c) 2020-2023 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>

#include <hpx/execution_base/stdexec_forward.hpp>
#include <type_traits>

namespace hpx::execution::experimental {
    template <typename Receiver>
    struct is_receiver : std::bool_constant<receiver<Receiver>>
    {
    };

    template <typename Receiver>
    inline constexpr bool is_receiver_v = is_receiver<Receiver>::value;

    template <typename Receiver, typename Completions>
    struct is_receiver_of
      : std::bool_constant<receiver_of<Receiver, Completions>>
    {
    };

    template <typename Receiver, typename Completions>
    inline constexpr bool is_receiver_of_v =
        is_receiver_of<Receiver, Completions>::value;

    namespace detail {

        // What about this implementation instead of using template specialization?
        template <typename CPO>
        struct is_receiver_cpo
          : std::bool_constant<std::is_same_v<CPO, set_value_t> ||
                std::is_same_v<CPO, set_error_t> ||
                std::is_same_v<CPO, set_stopped_t>>
        {
        };

        template <typename CPO>
        inline constexpr bool is_receiver_cpo_v = is_receiver_cpo<CPO>::value;
    }    // namespace detail

}    // namespace hpx::execution::experimental
