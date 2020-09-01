//  Copyright (c) 2007-2018 Hartmut Kaiser
//  Copyright (c) 2019 Agustin Berge
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/futures/future.hpp>

#include <hpx/executors/execution_policy.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace parallel { namespace util { namespace detail {
    template <typename ExPolicy, template <typename...> class Partitioner,
        template <typename...> class TaskPartitioner>
    struct select_partitioner
    {
        template <typename... Args>
        using apply = Partitioner<ExPolicy, Args...>;
    };

    template <template <typename...> class Partitioner,
        template <typename...> class TaskPartitioner>
    struct select_partitioner<hpx::execution::parallel_task_policy, Partitioner,
        TaskPartitioner>
    {
        template <typename... Args>
        using apply =
            TaskPartitioner<hpx::execution::parallel_task_policy, Args...>;
    };

    template <typename Executor, typename Parameters,
        template <typename...> class Partitioner,
        template <typename...> class TaskPartitioner>
    struct select_partitioner<
        hpx::execution::parallel_task_policy_shim<Executor, Parameters>,
        Partitioner, TaskPartitioner>
    {
        template <typename... Args>
        using apply = TaskPartitioner<
            hpx::execution::parallel_task_policy_shim<Executor, Parameters>,
            Args...>;
    };

#if defined(HPX_HAVE_DATAPAR)
    template <template <typename...> class Partitioner,
        template <typename...> class TaskPartitioner>
    struct select_partitioner<hpx::execution::datapar_task_policy, Partitioner,
        TaskPartitioner>
    {
        template <typename... Args>
        using apply =
            TaskPartitioner<hpx::execution::datapar_task_policy, Args...>;
    };
#endif
}}}}    // namespace hpx::parallel::util::detail
