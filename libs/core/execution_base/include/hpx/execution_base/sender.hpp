//  Copyright (c) 2020 Thomas Heller
//  Copyright (c) 2022-2023 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>

#include <hpx/execution_base/stdexec_forward.hpp>

namespace hpx::execution::experimental {
    template <typename Scheduler>
    inline constexpr bool is_scheduler_v = scheduler<Scheduler>;

    template <typename Scheduler>
    struct is_scheduler : std::bool_constant<is_scheduler_v<Scheduler>>
    {
    };
    // defined in completion signatures instead, to follow the original
    // file structure.
    namespace detail {
        // Dummy type used in place of a scheduler if none is given
        struct no_scheduler
        {
        };
    }    // namespace detail
}    // namespace hpx::execution::experimental
