//  Copyright (c)      2020 Mikael Simberg
//  Copyright (c) 2007-2017 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file parallel/executors/restricted_thread_pool_executors.hpp

#pragma once

#include <hpx/config.hpp>
#include <hpx/assert.hpp>
#include <hpx/execution/executors/execution_parameters.hpp>
#include <hpx/executors/thread_pool_executor.hpp>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <utility>
#include <vector>

namespace hpx { namespace parallel { namespace execution {
    class restricted_thread_pool_executor
    {
        static constexpr std::size_t hierarchical_threshold_default_ = 6;

    public:
        /// Associate the parallel_execution_tag executor tag type as a default
        /// with this executor.
        typedef hpx::execution::parallel_execution_tag execution_category;

        /// Associate the static_chunk_size executor parameters type as a default
        /// with this executor.
        typedef hpx::execution::static_chunk_size executor_parameters_type;

        /// Create a new parallel executor
        restricted_thread_pool_executor(std::size_t first_thread = 0,
            std::size_t num_threads = 1,
            threads::thread_priority priority =
                threads::thread_priority::default_,
            threads::thread_stacksize stacksize =
                threads::thread_stacksize::default_,
            threads::thread_schedule_hint schedulehint = {},
            std::size_t hierarchical_threshold =
                hierarchical_threshold_default_)
          : pool_(this_thread::get_pool())
          , priority_(priority)
          , stacksize_(stacksize)
          , schedulehint_(schedulehint)
          , hierarchical_threshold_(hierarchical_threshold)
          , first_thread_(first_thread)
          , num_threads_(num_threads)
          , os_thread_(first_thread_)
        {
            HPX_ASSERT(pool_);
        }

        restricted_thread_pool_executor(
            restricted_thread_pool_executor const& other)
          : pool_(other.pool_)
          , priority_(other.priority_)
          , stacksize_(other.stacksize_)
          , schedulehint_(other.schedulehint_)
          , first_thread_(other.first_thread_)
          , num_threads_(other.num_threads_)
          , os_thread_(other.first_thread_)
        {
            HPX_ASSERT(pool_);
        }

        /// \cond NOINTERNAL
        bool operator==(restricted_thread_pool_executor const& rhs) const
            noexcept
        {
            return pool_ == rhs.pool_ && priority_ == rhs.priority_ &&
                stacksize_ == rhs.stacksize_ &&
                schedulehint_ == rhs.schedulehint_ &&
                first_thread_ == rhs.first_thread_ &&
                num_threads_ == rhs.num_threads_;
        }

        bool operator!=(restricted_thread_pool_executor const& rhs) const
            noexcept
        {
            return !(*this == rhs);
        }

        restricted_thread_pool_executor const& context() const noexcept
        {
            return *this;
        }

    private:
        std::int16_t get_next_thread_num()
        {
            return static_cast<std::int16_t>(
                first_thread_ + (os_thread_++ % num_threads_));
        }

    public:
        template <typename F, typename... Ts>
        hpx::future<
            typename hpx::util::detail::invoke_deferred_result<F, Ts...>::type>
        async_execute(F&& f, Ts&&... ts)
        {
            return hpx::detail::async_launch_policy_dispatch<decltype(
                launch::async)>::call(launch::async, pool_, priority_,
                stacksize_,
                threads::thread_schedule_hint(get_next_thread_num()),
                std::forward<F>(f), std::forward<Ts>(ts)...);
        }

        template <typename F, typename Future, typename... Ts>
        hpx::future<typename hpx::util::detail::invoke_deferred_result<F,
            Future, Ts...>::type>
        then_execute(F&& f, Future&& predecessor, Ts&&... ts)
        {
            using result_type =
                typename hpx::util::detail::invoke_deferred_result<F, Future,
                    Ts...>::type;

            auto&& func = hpx::util::one_shot(hpx::util::bind_back(
                std::forward<F>(f), std::forward<Ts>(ts)...));

            typename hpx::traits::detail::shared_state_ptr<result_type>::type
                p = hpx::lcos::detail::make_continuation_exec<result_type>(
                    std::forward<Future>(predecessor), *this, std::move(func));

            return hpx::traits::future_access<
                hpx::lcos::future<result_type>>::create(std::move(p));
        }

        // NonBlockingOneWayExecutor (adapted) interface
        template <typename F, typename... Ts>
        void post(F&& f, Ts&&... ts)
        {
            char const* annotation =
                hpx::traits::get_function_annotation<F>::call(f);
            hpx::util::thread_description desc(f, annotation);

            detail::post_policy_dispatch<decltype(launch::async)>::call(
                launch::async, desc, pool_, priority_, stacksize_,
                threads::thread_schedule_hint(get_next_thread_num()),
                std::forward<F>(f), std::forward<Ts>(ts)...);
        }

        template <typename F, typename S, typename... Ts>
        std::vector<hpx::future<
            typename detail::bulk_function_result<F, S, Ts...>::type>>
        bulk_async_execute(F&& f, S const& shape, Ts&&... ts) const
        {
            return detail::hierarchical_bulk_async_execute_helper(pool_,
                priority_, stacksize_, schedulehint_, first_thread_,
                num_threads_, hierarchical_threshold_, launch::async,
                std::forward<F>(f), shape, std::forward<Ts>(ts)...);
        }

        template <typename F, typename S, typename Future, typename... Ts>
        hpx::future<typename detail::bulk_then_execute_result<F, S, Future,
            Ts...>::type>
        bulk_then_execute(
            F&& f, S const& shape, Future&& predecessor, Ts&&... ts)
        {
            return detail::hierarchical_bulk_then_execute_helper(*this,
                launch::async, std::forward<F>(f), shape,
                std::forward<Future>(predecessor), std::forward<Ts>(ts)...);
        }
        /// \endcond

    private:
        threads::thread_pool_base* pool_ = nullptr;

        threads::thread_priority priority_ = threads::thread_priority::default_;
        threads::thread_stacksize stacksize_ =
            threads::thread_stacksize::default_;
        threads::thread_schedule_hint schedulehint_ = {};
        std::size_t hierarchical_threshold_ = hierarchical_threshold_default_;

        std::size_t first_thread_;
        std::size_t num_threads_;
        std::atomic<std::size_t> os_thread_;
    };
}}}    // namespace hpx::parallel::execution

namespace hpx { namespace parallel { namespace execution {
    /// \cond NOINTERNAL
    template <>
    struct is_one_way_executor<
        parallel::execution::restricted_thread_pool_executor> : std::true_type
    {
    };

    template <>
    struct is_two_way_executor<
        parallel::execution::restricted_thread_pool_executor> : std::true_type
    {
    };

    template <>
    struct is_bulk_two_way_executor<
        parallel::execution::restricted_thread_pool_executor> : std::true_type
    {
    };
    /// \endcond
}}}    // namespace hpx::parallel::execution
