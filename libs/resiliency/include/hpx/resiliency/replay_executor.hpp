//  Copyright (c) 2020 Hartmut Kaiser
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
#include <hpx/execution/traits/is_executor.hpp>
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

namespace hpx { namespace resiliency { namespace experimental {

    ///////////////////////////////////////////////////////////////////////////
    template <typename BaseExecutor, typename Validate>
    class replay_executor
    {
    public:
        static constexpr int num_spread = 4;
        static constexpr int num_tasks = 128;

        using execution_category = typename BaseExecutor::execution_category;
        using executor_parameters_type =
            typename BaseExecutor::executor_parameters_type;

        template <typename Result>
        using future_type =
            typename hpx::parallel::execution::executor_future<BaseExecutor,
                Result>::type;

        template <typename F>
        explicit replay_executor(BaseExecutor& exec, std::size_t n, F&& f)
          : exec_(exec)
          , replay_count_(n)
          , validator_(std::forward<F>(f))
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

        replay_executor const& context() const noexcept
        {
            return *this;
        }

        // TwoWayExecutor interface
        template <typename F, typename... Ts>
        decltype(auto) async_execute(F&& f, Ts&&... ts) const
        {
            return async_replay_validate(exec_, replay_count_, validator_,
                std::forward<F>(f), std::forward<Ts>(ts)...);
        }

        // BulkTwoWayExecutor interface
        template <typename F, typename S, typename... Ts>
        decltype(auto) bulk_async_execute(
            F&& f, S const& shape, Ts&&... ts) const
        {
            std::size_t size = hpx::util::size(shape);

            using result_type =
                typename hpx::parallel::execution::detail::bulk_function_result<
                    F, S, Ts...>::type;
            using future_type =
                typename hpx::parallel::execution::executor_future<BaseExecutor,
                    result_type>::type;

            std::vector<future_type> results;
            results.resize(size);

            hpx::lcos::local::latch l(size + 1);

            spawn_hierarchical(results, l, 0, size, num_tasks, f,
                hpx::util::begin(shape), ts...);

            l.count_down_and_wait();

            return results;
        }

    protected:
        /// \cond NOINTERNAL
        template <typename Result, typename F, typename Iter, typename... Ts>
        void spawn_sequential(std::vector<hpx::future<Result>>& results,
            hpx::lcos::local::latch& l, std::size_t base, std::size_t size,
            F&& func, Iter it, Ts&&... ts) const
        {
            // spawn tasks sequentially
            HPX_ASSERT(base + size <= results.size());

            for (std::size_t i = 0; i != size; (void) ++i, ++it)
            {
                results[base + i] = async_execute(func, *it, ts...);
            }

            l.count_down(size);
        }

        template <typename Result, typename F, typename Iter, typename... Ts>
        void spawn_hierarchical(std::vector<hpx::future<Result>>& results,
            hpx::lcos::local::latch& l, std::size_t base, std::size_t size,
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
                        hpx::this_thread::get_executor(),
                        [&, base, chunk_size, num_tasks, it] {
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
    replay_executor<BaseExecutor, typename std::decay<Validate>::type>
    make_replay_executor(BaseExecutor& exec, std::size_t n, Validate&& validate)
    {
        return replay_executor<BaseExecutor,
            typename std::decay<Validate>::type>(
            exec, n, std::forward<Validate>(validate));
    }

    template <typename BaseExecutor>
    replay_executor<BaseExecutor, detail::replay_validator>
    make_replay_executor(BaseExecutor& exec, std::size_t n)
    {
        return replay_executor<BaseExecutor, detail::replay_validator>(
            exec, n, detail::replay_validator());
    }
}}}    // namespace hpx::resiliency::experimental

namespace hpx { namespace parallel { namespace execution {

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
}}}    // namespace hpx::parallel::execution
