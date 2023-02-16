//  Copyright (c) 2016-2023 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>

#if defined(HPX_HAVE_DATAPAR)
#include <hpx/executors/execution_policy_fwd.hpp>

// TODO: Should this be experimental?
namespace hpx::execution::detail {

    ///////////////////////////////////////////////////////////////////////////
    template <typename Executor, typename Parameters = void>
    struct simd_policy_shim;

    template <typename Executor, typename Parameters = void>
    struct simd_task_policy_shim;

    template <typename Executor, typename Parameters = void>
    struct par_simd_policy_shim;

    template <typename Executor, typename Parameters = void>
    struct par_simd_task_policy_shim;
}    // namespace hpx::execution::detail

#endif
