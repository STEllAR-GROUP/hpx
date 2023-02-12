//  Copyright (c) 2020-2023 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/resiliency/config.hpp>
#include <hpx/assert.hpp>
#include <hpx/async_base/launch_policy.hpp>
#include <hpx/execution/executors/execution.hpp>
#include <hpx/execution/traits/executor_traits.hpp>
#include <hpx/execution_base/traits/is_executor.hpp>
#include <hpx/executors/current_executor.hpp>
#include <hpx/futures/future.hpp>
#include <hpx/iterator_support/range.hpp>
#include <hpx/resiliency/async_replay_executor.hpp>
#include <hpx/synchronization/latch.hpp>

#include <algorithm>
#include <cstddef>
#include <type_traits>
#include <utility>
#include <vector>

namespace hpx::resiliency::experimental {

    ///////////////////////////////////////////////////////////////////////////
    template <typename BaseExecutor, typename Validate>
    class replay_executor
    {
    public:
        static constexpr int num_spread = 4;
        static constexpr int num_tasks = 128;

        using execution_category =
            hpx::traits::executor_execution_category_t<BaseExecutor>;
        using executor_parameters_type =
            hpx::traits::executor_parameters_type_t<BaseExecutor>;

        template <typename Result>
        using future_type =
            hpx::traits::executor_future_t<BaseExecutor, Result>;

        template <typename F>
        explicit replay_executor(BaseExecutor& exec, std::size_t n, F&& f)
          : exec_(exec)
          , replay_count_(n)
          , validator_(HPX_FORWARD(F, f))
        {
        }

        bool operator==(replay_executor const& rhs) const noexcept
        {
            return exec_ == rhs.exec_;
        }

        bool operator!=(replay_executor const& rhs) const noexcept
        {
            return !(*this == rhs);
        }

        constexpr replay_executor const& context() const noexcept
        {
            return *this;
        }

    private:
        // TwoWayExecutor interface
        template <typename F, typename... Ts>
        friend decltype(auto) tag_invoke(
            hpx::parallel::execution::async_execute_t,
            replay_executor const& exec, F&& f, Ts&&... ts)
        {
            return async_replay_validate(exec.exec_, exec.replay_count_,
                exec.validator_, HPX_FORWARD(F, f), HPX_FORWARD(Ts, ts)...);
        }

        // BulkTwoWayExecutor interface
        template <typename F, typename S, typename... Ts>
        friend decltype(auto) tag_invoke(
            hpx::parallel::execution::bulk_async_execute_t,
            replay_executor const& exec, F&& f, S const& shape, Ts&&... ts)
        {
            std::size_t size = hpx::util::size(shape);

            using result_type =
                hpx::parallel::execution::detail::bulk_function_result_t<F, S,
                    Ts...>;
            using future_type =
                hpx::traits::executor_future_t<BaseExecutor, result_type>;

            std::vector<future_type> results;
            results.resize(size);

            hpx::latch l(size + 1);

            exec.spawn_hierarchical(results, l, 0, size, num_tasks, f,
                hpx::util::begin(shape), ts...);

            l.arrive_and_wait();

            return results;
        }

    protected:
        /// \cond NOINTERNAL
        template <typename Result, typename F, typename Iter, typename... Ts>
        void spawn_sequential(std::vector<hpx::future<Result>>& results,
            hpx::latch& l, std::size_t base, std::size_t size, F&& func,
            Iter it, Ts&&... ts) const
        {
            // spawn tasks sequentially
            HPX_ASSERT(base + size <= results.size());

            for (std::size_t i = 0; i != size; (void) ++i, ++it)
            {
                results[base + i] =
                    tag_invoke(hpx::parallel::execution::async_execute_t{},
                        *this, func, *it, ts...);
            }

            l.count_down(size);
        }

        template <typename Result, typename F, typename Iter, typename... Ts>
        void spawn_hierarchical(std::vector<hpx::future<Result>>& results,
            hpx::latch& l, std::size_t base, std::size_t size,
            std::size_t num_tasks, F&& func, Iter it, Ts&&... ts) const
        {
            if (size > num_tasks)
            {
                // spawn hierarchical tasks
                std::size_t chunk_size = (size + num_spread) / num_spread - 1;
                chunk_size = (std::max)(chunk_size, num_tasks);

                while (size > chunk_size)
                {
                    hpx::parallel::execution::post(
                        exec_, [&, base, chunk_size, num_tasks, it] {
                            spawn_hierarchical(results, l, base, chunk_size,
                                num_tasks, func, it, ts...);
                        });

                    base += chunk_size;
                    std::advance(it, chunk_size);
                    size -= chunk_size;
                }
            }

            // spawn remaining tasks sequentially
            spawn_sequential(results, l, base, size, func, it, ts...);
        }
        /// \endcond

    private:
        BaseExecutor& exec_;
        std::size_t replay_count_;
        Validate validator_;
    };

    ///////////////////////////////////////////////////////////////////////////
    template <typename BaseExecutor, typename Validate>
    replay_executor<BaseExecutor, std::decay_t<Validate>> make_replay_executor(
        BaseExecutor& exec, std::size_t n, Validate&& validate)
    {
        return replay_executor<BaseExecutor, std::decay_t<Validate>>(
            exec, n, HPX_FORWARD(Validate, validate));
    }

    template <typename BaseExecutor>
    replay_executor<BaseExecutor, detail::replay_validator>
    make_replay_executor(BaseExecutor& exec, std::size_t n)
    {
        return replay_executor<BaseExecutor, detail::replay_validator>(
            exec, n, detail::replay_validator());
    }
}    // namespace hpx::resiliency::experimental

namespace hpx::parallel::execution {

    template <typename BaseExecutor, typename Validator>
    struct is_two_way_executor<
        hpx::resiliency::experimental::replay_executor<BaseExecutor, Validator>>
      : std::true_type
    {
    };

    template <typename BaseExecutor, typename Validator>
    struct is_bulk_two_way_executor<
        hpx::resiliency::experimental::replay_executor<BaseExecutor, Validator>>
      : std::true_type
    {
    };
}    // namespace hpx::parallel::execution
