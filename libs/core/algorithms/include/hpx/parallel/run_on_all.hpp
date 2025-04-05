//  Copyright (c) 2025 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file run_on_all.hpp
/// \page hpx::experimental::run_on_all
/// \headerfile hpx/experimental/run_on_all.hpp

#pragma once

#include <hpx/config.hpp>
#include <hpx/async_combinators/wait_all.hpp>
#include <hpx/concepts/concepts.hpp>
#include <hpx/execution/detail/execution_parameter_callbacks.hpp>
#include <hpx/execution/execution.hpp>
#include <hpx/execution_base/execution.hpp>
#include <hpx/executors/parallel_executor.hpp>
#include <hpx/functional/experimental/scope_exit.hpp>
#include <hpx/functional/invoke.hpp>
#include <hpx/parallel/algorithms/for_loop_reduction.hpp>

#include <cstddef>
#include <type_traits>
#include <utility>    // for std::move

namespace hpx::experimental {

    template <typename T, typename Op, typename F, typename... Ts>
    void run_on_all(std::size_t num_tasks,
        hpx::parallel::detail::reduction_helper<T, Op>&& r, F&& f, Ts&&... ts)
    {
        // force using index_queue scheduler with given amount of threads
        auto exec = hpx::execution::experimental::with_processing_units_count(
            hpx::execution::parallel_executor(
                hpx::threads::thread_priority::bound),
            num_tasks);
        exec.set_hierarchical_threshold(0);

        r.init_iteration(0, 0);
        auto on_exit =
            hpx::experimental::scope_exit([&] { r.exit_iteration(0); });

        hpx::wait_all(hpx::parallel::execution::bulk_async_execute(
            exec, [&](auto i) { f(r.iteration_value(i), ts...); }, num_tasks,
            HPX_FORWARD(Ts, ts)...));
    }

    template <typename T, typename Op, typename F, typename... Ts>
    void run_on_all(
        hpx::parallel::detail::reduction_helper<T, Op>&& r, F&& f, Ts&&... ts)
    {
        std::size_t cores =
            hpx::parallel::execution::detail::get_os_thread_count();
        run_on_all(
            cores, HPX_MOVE(r), HPX_FORWARD(F, f), HPX_FORWARD(Ts, ts)...);
    }

    template <typename F, typename... Ts>
    void run_on_all(std::size_t num_tasks, F&& f, Ts&&... ts)
    {
        // force using index_queue scheduler with given amount of threads
        auto exec = hpx::execution::experimental::with_processing_units_count(
            hpx::execution::parallel_executor(
                hpx::threads::thread_priority::bound),
            num_tasks);
        exec.set_hierarchical_threshold(0);

        hpx::wait_all(hpx::parallel::execution::bulk_async_execute(
            exec, [&](auto) { f(ts...); }, num_tasks, HPX_FORWARD(Ts, ts)...));
    }

    template <typename F, typename... Ts,
        HPX_CONCEPT_REQUIRES_(std::is_invocable_v<F&&, Ts&&...>)>
    void run_on_all(F&& f, Ts&&... ts)
    {
        std::size_t cores =
            hpx::parallel::execution::detail::get_os_thread_count();
        run_on_all(cores, HPX_FORWARD(F, f), HPX_FORWARD(Ts, ts)...);
    }

    // Execution policy overloads
    template <typename ExPolicy, typename T, typename Op, typename F,
        typename... Ts,
        HPX_CONCEPT_REQUIRES_(hpx::is_execution_policy_v<ExPolicy>)>
    auto run_on_all(ExPolicy&& policy, T&& init, Op&& op, F&& f, Ts&&... ts)
    {
        static_assert(hpx::is_sequenced_execution_policy_v<ExPolicy> ||
                hpx::is_parallel_execution_policy_v<ExPolicy>,
            "hpx::is_sequenced_execution_policy_v<ExPolicy> || "
            "hpx::is_parallel_execution_policy_v<ExPolicy>");

        using result_type = hpx::util::invoke_result_t<Op, T, T>;
        using future_type = hpx::future<result_type>;

        if constexpr (hpx::is_async_execution_policy_v<ExPolicy>)
        {
            return hpx::async(policy.executor(),
                [init = std::forward<T>(init), op = std::forward<Op>(op),
                    f = std::forward<F>(f),
                    ... ts = std::forward<Ts>(ts)]() mutable {
                    return run_on_all_impl(std::move(init), std::move(op),
                        std::move(f), std::move(ts)...);
                });
        }
        else
        {
            return run_on_all_impl(std::forward<T>(init), std::forward<Op>(op),
                std::forward<F>(f), std::forward<Ts>(ts)...);
        }
    }

    template <typename ExPolicy, typename T, typename Op, typename F,
        typename... Ts,
        HPX_CONCEPT_REQUIRES_(hpx::is_execution_policy_v<ExPolicy>)>
    auto run_on_all(ExPolicy&& policy,
        hpx::parallel::detail::reduction_helper<T, Op>&& r, F&& f, Ts&&... ts)
    {
        std::size_t cores =
            hpx::parallel::execution::detail::get_os_thread_count();
        return run_on_all(HPX_FORWARD(ExPolicy, policy), cores, HPX_MOVE(r),
            HPX_FORWARD(F, f), HPX_FORWARD(Ts, ts)...);
    }

    template <typename ExPolicy, typename F, typename... Ts,
        HPX_CONCEPT_REQUIRES_(hpx::is_execution_policy_v<ExPolicy>)>
    auto run_on_all(ExPolicy&& policy, std::size_t num_tasks, F&& f, Ts&&... ts)
    {
        // force using index_queue scheduler with given amount of threads
        auto exec = hpx::execution::experimental::with_processing_units_count(
            hpx::execution::parallel_executor(
                hpx::threads::thread_priority::bound),
            num_tasks);
        exec.set_hierarchical_threshold(0);

        return hpx::parallel::execution::bulk_async_execute(
            exec, [&](auto) { f(ts...); }, num_tasks, HPX_FORWARD(Ts, ts)...);
    }

    template <typename ExPolicy, typename F, typename... Ts,
        HPX_CONCEPT_REQUIRES_(hpx::is_execution_policy_v<ExPolicy>&&
                std::is_invocable_v<F&&, Ts&&...>)>
    auto run_on_all(ExPolicy&& policy, F&& f, Ts&&... ts)
    {
        std::size_t cores =
            hpx::parallel::execution::detail::get_os_thread_count();
        return run_on_all(HPX_FORWARD(ExPolicy, policy), cores,
            HPX_FORWARD(F, f), HPX_FORWARD(Ts, ts)...);
    }
}    // namespace hpx::experimental
