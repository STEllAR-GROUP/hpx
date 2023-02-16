//  Copyright (c) 2007-2023 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>

namespace hpx::execution::detail {

    // forward declarations, see execution_policy.hpp
    template <typename Executor, typename Parameters = void>
    struct sequenced_policy_shim;

    template <typename Executor, typename Parameters = void>
    struct sequenced_task_policy_shim;

    template <typename Executor, typename Parameters = void>
    struct parallel_policy_shim;

    template <typename Executor, typename Parameters = void>
    struct parallel_task_policy_shim;

    template <typename Executor, typename Parameters = void>
    struct unsequenced_task_policy_shim;

    template <typename Executor, typename Parameters = void>
    struct unsequenced_policy_shim;

    template <typename Executor, typename Parameters = void>
    struct parallel_unsequenced_task_policy_shim;

    template <typename Executor, typename Parameters = void>
    struct parallel_unsequenced_policy_shim;
}    // namespace hpx::execution::detail
