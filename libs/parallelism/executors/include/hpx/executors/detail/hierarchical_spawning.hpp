//  Copyright (c) 2019-2020 ETH Zurich
//  Copyright (c) 2007-2019 Hartmut Kaiser
//  Copyright (c) 2019 Agustin Berge
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/assert.hpp>
#include <hpx/async_base/launch_policy.hpp>
#include <hpx/coroutines/thread_enums.hpp>
#include <hpx/execution/algorithms/detail/predicates.hpp>
#include <hpx/execution/detail/async_launch_policy_dispatch.hpp>
#include <hpx/execution/detail/post_policy_dispatch.hpp>
#include <hpx/execution/executors/execution.hpp>
#include <hpx/execution/executors/fused_bulk_execute.hpp>
#include <hpx/futures/future.hpp>
#include <hpx/futures/traits/future_traits.hpp>
#include <hpx/iterator_support/range.hpp>
#include <hpx/pack_traversal/unwrap.hpp>
#include <hpx/synchronization/latch.hpp>
#include <hpx/threading_base/scheduler_base.hpp>
#include <hpx/threading_base/thread_data.hpp>
#include <hpx/threading_base/thread_helpers.hpp>
#include <hpx/threading_base/thread_pool_base.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <utility>
#include <vector>

namespace hpx { namespace parallel { namespace execution { namespace detail {
    template <typename F, typename S, typename... Ts>
    std::vector<
        hpx::future<typename detail::bulk_function_result<F, S, Ts...>::type>>
    hierarchical_bulk_async_execute_helper(threads::thread_pool_base* pool,
        threads::thread_priority priority, threads::thread_stacksize stacksize,
        threads::thread_schedule_hint, std::size_t first_thread,
        std::size_t num_threads, std::size_t hierarchical_threshold,
        launch policy, F&& f, S const& shape, Ts&&... ts)
    {
        HPX_ASSERT(pool);

        hpx::util::thread_description const desc(f,
            "hpx::parallel::execution::detail::hierarchical_bulk_async_execute_"
            "helper");

        typedef std::vector<hpx::future<
            typename detail::bulk_function_result<F, S, Ts...>::type>>
            result_type;

        result_type results;
        std::size_t const size = hpx::util::size(shape);
        results.resize(size);

        lcos::local::latch l(size);
        std::size_t part_begin = 0;
        auto it = std::begin(shape);
        for (std::size_t t = 0; t < num_threads; ++t)
        {
            std::size_t const part_end = ((t + 1) * size) / num_threads;
            std::size_t const part_size = part_end - part_begin;

            threads::thread_schedule_hint hint{
                static_cast<std::int16_t>(first_thread + t)};

            if (part_size > hierarchical_threshold)
            {
                detail::post_policy_dispatch<decltype(policy)>::call(policy,
                    desc, pool, priority, threads::thread_stacksize::small_,
                    hint,
                    [&, hint, part_begin, part_end, part_size, f,
                        it]() mutable {
                        for (std::size_t part_i = part_begin; part_i < part_end;
                             ++part_i)
                        {
                            results[part_i] =
                                hpx::detail::async_launch_policy_dispatch<
                                    decltype(policy)>::call(policy, pool,
                                    priority, stacksize, hint, f, *it, ts...);
                            ++it;
                        }
                        l.count_down(part_size);
                    });

                std::advance(it, part_size);
            }
            else
            {
                for (std::size_t part_i = part_begin; part_i < part_end;
                     ++part_i)
                {
                    results[part_i] =
                        hpx::detail::async_launch_policy_dispatch<decltype(
                            policy)>::call(policy, pool, priority, stacksize,
                            hint, f, *it, ts...);
                    ++it;
                }
                l.count_down(part_size);
            }

            part_begin = part_end;
        }

        l.wait();

        return results;
    }

    template <typename Executor, typename F, typename S, typename Future,
        typename... Ts>
    hpx::future<
        typename detail::bulk_then_execute_result<F, S, Future, Ts...>::type>
    hierarchical_bulk_then_execute_helper(Executor&& executor, launch policy,
        F&& f, S const& shape, Future&& predecessor, Ts&&... ts)
    {
        using func_result_type = typename detail::then_bulk_function_result<F,
            S, Future, Ts...>::type;

        // std::vector<future<func_result_type>>
        using result_type = std::vector<hpx::future<func_result_type>>;

        auto&& func = detail::make_fused_bulk_async_execute_helper<result_type>(
            executor, std::forward<F>(f), shape,
            hpx::make_tuple(std::forward<Ts>(ts)...));

        // void or std::vector<func_result_type>
        using vector_result_type = typename detail::bulk_then_execute_result<F,
            S, Future, Ts...>::type;

        // future<vector_result_type>
        using result_future_type = hpx::future<vector_result_type>;

        using shared_state_type =
            typename hpx::traits::detail::shared_state_ptr<
                vector_result_type>::type;

        using future_type = typename std::decay<Future>::type;

        // vector<future<func_result_type>> -> vector<func_result_type>
        shared_state_type p = hpx::lcos::detail::make_continuation_exec_policy<
            vector_result_type>(std::forward<Future>(predecessor), executor,
            policy,
            [func = std::move(func)](
                future_type&& predecessor) mutable -> vector_result_type {
                // use unwrap directly (instead of lazily) to avoid
                // having to pull in dataflow
                return hpx::util::unwrap(func(std::move(predecessor)));
            });

        return hpx::traits::future_access<result_future_type>::create(
            std::move(p));
    }
}}}}    // namespace hpx::parallel::execution::detail
