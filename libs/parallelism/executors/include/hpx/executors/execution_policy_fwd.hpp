//  Copyright (c) 2007-2015 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#if defined(HPX_HAVE_DATAPAR)
#include <hpx/executors/datapar/execution_policy_fwd.hpp>
#endif

namespace hpx { namespace execution {
    ///////////////////////////////////////////////////////////////////////////
    // forward declarations, see execution_policy.hpp
    struct sequenced_policy;

    template <typename Executor, typename Parameters>
    struct sequenced_policy_shim;

    struct sequenced_task_policy;

    template <typename Executor, typename Parameters>
    struct sequenced_task_policy_shim;

    struct parallel_policy;

    template <typename Executor, typename Parameters>
    struct parallel_policy_shim;

    struct parallel_task_policy;

    template <typename Executor, typename Parameters>
    struct parallel_task_policy_shim;

    struct parallel_unsequenced_policy;
}}    // namespace hpx::execution
