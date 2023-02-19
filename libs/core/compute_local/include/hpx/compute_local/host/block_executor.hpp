//  Copyright (c) 2016 Thomas Heller
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/async_combinators/when_all.hpp>
#include <hpx/compute_local/host/target.hpp>
#include <hpx/execution/executors/execution.hpp>
#include <hpx/execution/executors/static_chunk_size.hpp>
#include <hpx/execution/traits/executor_traits.hpp>
#include <hpx/execution_base/traits/is_executor.hpp>
#include <hpx/executors/restricted_thread_pool_executor.hpp>
#include <hpx/functional/deferred_call.hpp>
#include <hpx/futures/future.hpp>
#include <hpx/iterator_support/iterator_range.hpp>
#include <hpx/iterator_support/range.hpp>
#include <hpx/pack_traversal/unwrap.hpp>

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <exception>
#include <iterator>
#include <type_traits>
#include <utility>
#include <vector>

namespace hpx { namespace compute { namespace host {

    /// The block executor can be used to build NUMA aware programs.
    /// It will distribute work evenly across the passed targets
    ///
    /// \tparam Executor The underlying executor to use
    template <typename Executor =
                  hpx::parallel::execution::restricted_thread_pool_executor>
    struct block_executor
    {
    public:
        using executor_parameters_type =
            hpx::execution::experimental::static_chunk_size;

        explicit block_executor(std::vector<host::target> const& targets,
            threads::thread_priority priority = threads::thread_priority::high,
            threads::thread_stacksize stacksize =
                threads::thread_stacksize::default_,
            threads::thread_schedule_hint schedulehint = {})
          : targets_(targets)
          , current_(0)
          , priority_(priority)
          , stacksize_(stacksize)
          , schedulehint_(schedulehint)
        {
            init_executors();
        }

        explicit block_executor(std::vector<host::target>&& targets)
          : targets_(HPX_MOVE(targets))
          , current_(0)
        {
            init_executors();
        }

        block_executor(block_executor const& other)
          : targets_(other.targets_)
          , current_(0)
          , executors_(other.executors_)
        {
        }

        block_executor(block_executor&& other) noexcept
          : targets_(HPX_MOVE(other.targets_))
          , current_(other.current_.load())
          , executors_(HPX_MOVE(other.executors_))
        {
        }

        block_executor& operator=(block_executor const& other)
        {
            if (&other != this)
            {
                targets_ = other.targets_;
                current_ = 0;
                executors_ = other.executors_;
            }
            return *this;
        }

        block_executor& operator=(block_executor&& other) noexcept
        {
            if (&other != this)
            {
                targets_ = HPX_MOVE(other.targets_);
                current_ = other.current_.load();
                executors_ = HPX_MOVE(other.executors_);
            }
            return *this;
        }

        /// \cond NOINTERNAL
        bool operator==(block_executor const& rhs) const noexcept
        {
            return std::equal(
                targets_.begin(), targets_.end(), rhs.targets_.begin());
        }

        bool operator!=(block_executor const& rhs) const noexcept
        {
            return !(*this == rhs);
        }

        std::vector<host::target> const& context() const noexcept
        {
            return targets_;
        }
        /// \endcond

    private:
        // this function is conceptually const (current_ is mutable)
        auto get_next_executor() const
        {
            return executors_[current_++ % executors_.size()];
        }

        template <typename F, typename... Ts>
        friend decltype(auto) tag_invoke(hpx::parallel::execution::post_t,
            block_executor const& exec, F&& f, Ts&&... ts)
        {
            hpx::parallel::execution::post(exec.get_next_executor(),
                HPX_FORWARD(F, f), HPX_FORWARD(Ts, ts)...);
        }

        template <typename F, typename... Ts>
        friend decltype(auto) tag_invoke(
            hpx::parallel::execution::async_execute_t,
            block_executor const& exec, F&& f, Ts&&... ts)
        {
            return hpx::parallel::execution::async_execute(
                exec.get_next_executor(), HPX_FORWARD(F, f),
                HPX_FORWARD(Ts, ts)...);
        }

        template <typename F, typename... Ts>
        friend decltype(auto) tag_invoke(
            hpx::parallel::execution::sync_execute_t,
            block_executor const& exec, F&& f, Ts&&... ts)
        {
            return hpx::parallel::execution::sync_execute(
                exec.get_next_executor(), HPX_FORWARD(F, f),
                HPX_FORWARD(Ts, ts)...);
        }

        template <typename F, typename Shape, typename... Ts>
        decltype(auto) bulk_async_execute_impl(
            F&& f, Shape const& shape, Ts&&... ts) const
        {
            using result_type =
                parallel::execution::detail::bulk_function_result_t<F, Shape,
                    Ts...>;

            std::vector<hpx::future<result_type>> results;
            std::size_t cnt = util::size(shape);
            std::size_t num_executors = executors_.size();

            results.reserve(cnt);

            try
            {
                auto begin = util::begin(shape);
                for (std::size_t i = 0; i != executors_.size(); ++i)
                {
                    std::size_t part_begin_offset = (i * cnt) / num_executors;
                    std::size_t part_end_offset =
                        ((i + 1) * cnt) / num_executors;
                    auto part_begin = begin;
                    auto part_end = begin;
                    std::advance(part_begin, part_begin_offset);
                    std::advance(part_end, part_end_offset);
                    auto futures = hpx::parallel::execution::bulk_async_execute(
                        executors_[i], HPX_FORWARD(F, f),
                        util::iterator_range(part_begin, part_end),
                        HPX_FORWARD(Ts, ts)...);

                    if constexpr (hpx::traits::is_future_v<decltype(futures)>)
                    {
                        results.push_back(HPX_MOVE(futures));
                    }
                    else
                    {
                        results.insert(results.end(),
                            std::make_move_iterator(futures.begin()),
                            std::make_move_iterator(futures.end()));
                    }
                }
            }
            catch (std::bad_alloc const& ba)
            {
                throw ba;
            }
            catch (...)
            {
                results.clear();
                results.emplace_back(hpx::make_exceptional_future<result_type>(
                    std::current_exception()));
            }
            return results;
        }

        template <typename F, typename Shape, typename... Ts>
        friend decltype(auto) tag_invoke(
            hpx::parallel::execution::bulk_async_execute_t,
            block_executor const& exec, F&& f, Shape const& shape, Ts&&... ts)
        {
            return exec.bulk_async_execute_impl(
                HPX_FORWARD(F, f), shape, HPX_FORWARD(Ts, ts)...);
        }

        template <typename F, typename Shape, typename... Ts>
        decltype(auto) bulk_sync_execute_impl(
            F&& f, Shape const& shape, Ts&&... ts) const
        {
            using result_type =
                hpx::parallel::execution::detail::bulk_execute_result_t<F,
                    Shape, Ts...>;

            std::vector<result_type> results;
            std::size_t cnt = util::size(shape);
            std::size_t num_executors = executors_.size();

            results.reserve(cnt);

            try
            {
                auto begin = util::begin(shape);
                for (std::size_t i = 0; i != num_executors; ++i)
                {
                    std::size_t part_begin_offset = (i * cnt) / num_executors;
                    std::size_t part_end_offset =
                        ((i + 1) * cnt) / num_executors;
                    auto part_begin = begin;
                    auto part_end = begin;
                    std::advance(part_begin, part_begin_offset);
                    std::advance(part_end, part_end_offset);
                    auto part_results =
                        hpx::parallel::execution::bulk_sync_execute(
                            executors_[i], HPX_FORWARD(F, f),
                            util::iterator_range(begin, part_end),
                            HPX_FORWARD(Ts, ts)...);
                    results.emplace(results.end(),
                        std::make_move_iterator(part_results.begin()),
                        std::make_move_iterator(part_results.end()));
                }
                return results;
            }
            catch (std::bad_alloc const& ba)
            {
                throw ba;
            }
            catch (...)
            {
                throw exception_list(std::current_exception());
            }
        }

        template <typename F, typename Shape, typename... Ts>
        friend decltype(auto) tag_invoke(
            hpx::parallel::execution::bulk_sync_execute_t,
            block_executor const& exec, F&& f, Shape const& shape, Ts&&... ts)
        {
            return exec.bulk_sync_execute_impl(
                HPX_FORWARD(F, f), shape, HPX_FORWARD(Ts, ts)...);
        }

    public:
        std::vector<host::target> const& targets() const
        {
            return targets_;
        }

    private:
        void init_executors()
        {
            executors_.reserve(targets_.size());
            for (auto const& tgt : targets_)
            {
                auto num_pus = tgt.num_pus();
                executors_.emplace_back(num_pus.first, num_pus.second,
                    priority_, stacksize_, schedulehint_);
            }
        }

        std::vector<host::target> targets_;
        mutable std::atomic<std::size_t> current_;
        std::vector<Executor> executors_;
        threads::thread_priority priority_ = threads::thread_priority::high;
        threads::thread_stacksize stacksize_ =
            threads::thread_stacksize::default_;
        threads::thread_schedule_hint schedulehint_ = {};
    };
}}}    // namespace hpx::compute::host

namespace hpx { namespace parallel { namespace execution {
    template <typename Executor>
    struct executor_execution_category<compute::host::block_executor<Executor>>
    {
        typedef hpx::execution::parallel_execution_tag type;
    };

    template <typename Executor>
    struct is_one_way_executor<compute::host::block_executor<Executor>>
      : std::true_type
    {
    };

    template <typename Executor>
    struct is_two_way_executor<compute::host::block_executor<Executor>>
      : std::true_type
    {
    };

    template <typename Executor>
    struct is_bulk_one_way_executor<compute::host::block_executor<Executor>>
      : std::true_type
    {
    };

    template <typename Executor>
    struct is_bulk_two_way_executor<compute::host::block_executor<Executor>>
      : std::true_type
    {
    };
}}}    // namespace hpx::parallel::execution
