//  Copyright (c) 2017-2018 John Biddiscombe
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/execution/traits/executor_traits.hpp>
#include <hpx/execution/traits/is_executor.hpp>
#include <hpx/execution_base/execution.hpp>
#include <hpx/execution_base/this_thread.hpp>
#include <hpx/functional/invoke.hpp>
#include <hpx/futures/futures_factory.hpp>
#include <hpx/runtime/threads/thread_pool_base.hpp>
#include <hpx/type_support/void_guard.hpp>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <string>
#include <type_traits>
#include <utility>

#include <hpx/config/warnings_prefix.hpp>

// --------------------------------------------------------------------
//
// --------------------------------------------------------------------
namespace hpx { namespace parallel { namespace execution {
    template <typename BaseExecutor>
    struct limiting_executor
    {
      private:
        // --------------------------------------------------------------------
        // RAII wrapper for counting task completions (count_down)
        // count_up is done in the executor when the task is first scheduled
        // This object is contructed when the task actually _runs_
        // and destructed when it completes
        // --------------------------------------------------------------------
        struct on_exit
        {
            on_exit(limiting_executor const& this_e)
              : executor_(this_e)
            {
            }
            ~on_exit()
            {
                executor_.count_down();
            }
            limiting_executor const& executor_;
        };

        template <typename F>
        struct executor_wrapper
        {
            template <typename... Ts>
            decltype(auto) operator()(Ts&&... ts)
            {
                on_exit _{executor_};
                return hpx::util::invoke(f_, std::forward<Ts>(ts)...);
            }

            limiting_executor const& executor_;
            F f_;
        };

    public:
        using execution_category = typename BaseExecutor::execution_category;
        using executor_parameters_type =
            typename BaseExecutor::executor_parameters_type;

        // --------------------------------------------------------------------
        limiting_executor(BaseExecutor &ex, std::size_t lower, std::size_t upper,
            bool block_on_destruction = true)
          : executor_(ex)
          , count_(0)
          , lower_threshold_(lower)
          , upper_threshold_(upper)
          , block_(block_on_destruction)
        {
        }

        limiting_executor(std::size_t lower, std::size_t upper,
            bool block_on_destruction = true)
          : executor_(BaseExecutor{})
          , count_(0)
          , lower_threshold_(lower)
          , upper_threshold_(upper)
          , block_(block_on_destruction)
        {
        }

        // --------------------------------------------------------------------
        ~limiting_executor()
        {
            if (block_)
            {
                set_and_wait(0, 0);
            }
        }

        // --------------------------------------------------------------------
        limiting_executor const& context() const noexcept
        {
            return *this;
        }

        // --------------------------------------------------------------------
        // OneWayExecutor interface
        template <typename F, typename... Ts>
        decltype(auto) sync_execute(F&& f, Ts&&... ts) const
        {
            count_up();
            return hpx::parallel::execution::sync_execute(executor_,
                executor_wrapper<F>{*this, std::forward<F>(f)},
                std::forward<Ts>(ts)...);
        }

        // --------------------------------------------------------------------
        // TwoWayExecutor interface
        template <typename F, typename... Ts>
        decltype(auto) async_execute(F&& f, Ts&&... ts) const
        {
            count_up();
            return hpx::parallel::execution::async_execute(executor_,
                executor_wrapper<F>{*this, std::forward<F>(f)},
                std::forward<Ts>(ts)...);
        }

        template <typename F, typename Future, typename... Ts>
        decltype(auto) then_execute(
            F&& f, Future&& predecessor, Ts&&... ts) const
        {
            count_up();
            return hpx::parallel::execution::then_execute(executor_,
                executor_wrapper<F>{*this, std::forward<F>(f)},
                std::forward<Future>(predecessor), std::forward<Ts>(ts)...);
        }

        // --------------------------------------------------------------------
        // post : for general apply()
        // NonBlockingOneWayExecutor (adapted) interface
        // --------------------------------------------------------------------
        template <typename F, typename... Ts>
        void post(F&& f, Ts&&... ts)
        {
            count_up();
            hpx::parallel::execution::post(executor_,
                executor_wrapper<F>{*this, std::forward<F>(f)},
                std::forward<Ts>(ts)...);
        }

        // --------------------------------------------------------------------
        // BulkTwoWayExecutor interface
        template <typename F, typename S, typename... Ts>
        decltype(auto) bulk_async_execute(
            F&& f, S const& shape, Ts&&... ts) const
        {
            count_up();
            return hpx::parallel::execution::bulk_async_execute(executor_,
                executor_wrapper<F>{*this, std::forward<F>(f)}, shape,
                std::forward<Ts>(ts)...);
        }

        // --------------------------------------------------------------------
        template <typename F, typename S, typename Future, typename... Ts>
        decltype(auto) bulk_then_execute(
            F&& f, S const& shape, Future&& predecessor, Ts&&... ts) const
        {
            count_up();
            return hpx::parallel::execution::bulk_then_execute(executor_,
                executor_wrapper<F>{*this, std::forward<F>(f)}, shape,
                std::forward<Future>(predecessor), std::forward<Ts>(ts)...);
        }

        // --------------------------------------------------------------------
        // wait (suspend) until the number of tasks 'in flight' on this executor
        // drops to the lower threashold
        void wait()
        {
            hpx::util::yield_while(
                [&]() { return (count_ > lower_threshold_); });
        }

        // --------------------------------------------------------------------
        // wait (suspend) until all tasks launched on this executor have completed
        void wait_all()
        {
            hpx::util::yield_while([&]() { return (count_ > 0); });
        }

        void set_threshold(std::size_t lower, std::size_t upper)
        {
            lower_threshold_ = lower;
            upper_threshold_ = upper;
        }

      private:
        void count_up() const
        {
            if (++count_ > upper_threshold_)
            {
                hpx::util::yield_while(
                    [&]() { return (count_ > lower_threshold_); });
            }
        }

        void count_down() const
        {
            --count_;
        }

        void set_and_wait(std::size_t lower, std::size_t upper)
        {
            set_threshold(lower, upper);
            wait();
        }

    private:
        // --------------------------------------------------------------------
        BaseExecutor executor_;
        mutable std::atomic<std::int64_t> count_;
        mutable std::int64_t lower_threshold_;
        mutable std::int64_t upper_threshold_;
        bool block_;
    };

    // --------------------------------------------------------------------
    // simple forwarding implementations of executor traits
    // --------------------------------------------------------------------
    template <typename BaseExecutor>
    struct is_one_way_executor<
        limiting_executor<BaseExecutor>>
      : is_one_way_executor<typename std::decay<BaseExecutor>::type>
    {
    };

    template <typename BaseExecutor>
    struct is_never_blocking_one_way_executor<
        limiting_executor<BaseExecutor>>
      : is_never_blocking_one_way_executor<
            typename std::decay<BaseExecutor>::type>
    {
    };

    template <typename BaseExecutor>
    struct is_two_way_executor<
        limiting_executor<BaseExecutor>>
      : is_two_way_executor<typename std::decay<BaseExecutor>::type>
    {
    };

    template <typename BaseExecutor>
    struct is_bulk_one_way_executor<
        limiting_executor<BaseExecutor>>
      : is_bulk_one_way_executor<typename std::decay<BaseExecutor>::type>
    {
    };

    template <typename BaseExecutor>
    struct is_bulk_two_way_executor<
        limiting_executor<BaseExecutor>>
      : is_bulk_two_way_executor<typename std::decay<BaseExecutor>::type>
    {
    };
}}}    // namespace hpx::parallel::execution


#include <hpx/config/warnings_suffix.hpp>
