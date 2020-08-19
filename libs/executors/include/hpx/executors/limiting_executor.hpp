//  Copyright (c) 2017-2018 John Biddiscombe
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/concepts/has_member_xxx.hpp>
#include <hpx/execution/traits/executor_traits.hpp>
#include <hpx/execution/traits/is_executor.hpp>
#include <hpx/execution_base/execution.hpp>
#include <hpx/execution_base/this_thread.hpp>
#include <hpx/functional/invoke.hpp>
#include <hpx/threading_base/print.hpp>

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
namespace hpx { namespace execution { namespace experimental {

    // by convention the title is 7 chars (for alignment)
    using print_on = hpx::debug::enable_print<false>;
    static constexpr print_on lim_debug("LIMEXEC");

    ///////////////////////////////////////////////////////////////////////////
    namespace detail {
        HPX_HAS_MEMBER_XXX_TRAIT_DEF(in_flight_estimate)
    }    // namespace detail

    template <typename BaseExecutor>
    struct limiting_executor
    {
        // --------------------------------------------------------------------
        // RAII wrapper for counting task completions (count_down)
        // count_up is done in the executor when the task is first scheduled
        // This object is destructed when the task has completed
        // --------------------------------------------------------------------
        struct on_exit
        {
            on_exit(limiting_executor const& this_e)
              : executor_(this_e)
            {
            }
            ~on_exit()
            {
                lim_debug.debug(hpx::debug::str<>("Count Down"));
                executor_.count_down();
            }
            limiting_executor const& executor_;
        };

        // --------------------------------------------------------------------
        // this is the default wrapper struct that invokes count up / down
        // and uses the limiting executor counter to control throttling
        //
        // Note that we have to add a dummy template parameter B (same as BaseName)
        // to inner struct, to allow template deduction to SFINAE properly
        // and not give us an incomplete type on the 'in flight' specialization
        // --------------------------------------------------------------------
        template <typename F, typename B = BaseExecutor, typename Enable = void>
        struct throttling_wrapper
        {
            throttling_wrapper(
                limiting_executor& lim, BaseExecutor const& base, F&& f)
              : limiting_(lim)
              , f_(std::forward<F>(f))
            {
                limiting_.count_up();
                if (exceeds_upper())
                {
                    lim_debug.debug(hpx::debug::str<>("Exceeds_upper"));
                    hpx::util::yield_while([&]() { return exceeds_lower(); });
                    lim_debug.debug(hpx::debug::str<>("Below_lower"));
                }
            }

            // when task completes, on_exit destructor calls count_down
            template <typename... Ts>
            decltype(auto) operator()(Ts&&... ts)
            {
                on_exit _{limiting_};
                return hpx::util::invoke(f_, std::forward<Ts>(ts)...);
            }

            // returns true if too many tasks would be in flight
            // NB. we use ">" because we count up right before testing
            bool exceeds_upper()
            {
                return (limiting_.count_ > limiting_.upper_threshold_);
            }
            // returns true if we have not yet reached the lower threshold
            bool exceeds_lower()
            {
                return (limiting_.count_ > limiting_.lower_threshold_);
            }

            limiting_executor& limiting_;
            F f_;
        };

        // this is a specialized wrapper struct that skips count up / down
        // and uses the underlying executor to get a count of tasks
        // 'in flight' for the throttling.
        // The dummy B template param must match BaseExecutor and helps
        // deduction rules complete this type so that the default implementaiton
        // (above) still works
        template <typename F, typename B>
        struct throttling_wrapper<F, B,
            typename std::enable_if<
                detail::has_in_flight_estimate<BaseExecutor>::value &&
                std::is_same<B, BaseExecutor>::value>::type>
        {
            throttling_wrapper(
                limiting_executor const& lim, BaseExecutor const& base, F&& f)
              : limiting_(lim)
              , f_(std::forward<F>(f))
            {
                if (exceeds_upper(base))
                {
                    lim_debug.debug(hpx::debug::str<>("Exceeds_upper"),
                        "in_flight",
                        hpx::debug::dec<4>(base.in_flight_estimate()));
                    hpx::util::yield_while(
                        [&]() { return exceeds_lower(base); });
                    lim_debug.debug(hpx::debug::str<>("Below_lower"),
                        "in_flight",
                        hpx::debug::dec<4>(base.in_flight_estimate()));
                }
            }

            template <typename... Ts>
            decltype(auto) operator()(Ts&&... ts)
            {
                return hpx::util::invoke(f_, std::forward<Ts>(ts)...);
            }

            // NB. use ">=" because counting is external
            // (after invocation probably)
            bool exceeds_upper(BaseExecutor const& base) const
            {
                return (
                    base.in_flight_estimate() >= limiting_.upper_threshold_);
            }
            bool exceeds_lower(BaseExecutor const& base) const
            {
                return (base.in_flight_estimate() > limiting_.lower_threshold_);
            }

            limiting_executor const& limiting_;
            F f_;
        };

    public:
        using execution_category = typename BaseExecutor::execution_category;
        using executor_parameters_type =
            typename BaseExecutor::executor_parameters_type;

        // --------------------------------------------------------------------
        limiting_executor(BaseExecutor& ex, std::size_t lower,
            std::size_t upper, bool block_on_destruction = true)
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
            return hpx::parallel::execution::sync_execute(executor_,
                throttling_wrapper<F>(*this, executor_, std::forward<F>(f)),
                std::forward<Ts>(ts)...);
        }

        // --------------------------------------------------------------------
        // TwoWayExecutor interface
        template <typename F, typename... Ts>
        decltype(auto) async_execute(F&& f, Ts&&... ts)
        {
            return hpx::parallel::execution::async_execute(executor_,
                throttling_wrapper<F>(*this, executor_, std::forward<F>(f)),
                std::forward<Ts>(ts)...);
        }

        template <typename F, typename Future, typename... Ts>
        decltype(auto) then_execute(F&& f, Future&& predecessor, Ts&&... ts)
        {
            return hpx::parallel::execution::then_execute(executor_,
                throttling_wrapper<F>(*this, executor_, std::forward<F>(f)),
                std::forward<Future>(predecessor), std::forward<Ts>(ts)...);
        }

        // --------------------------------------------------------------------
        // post : for general apply()
        // NonBlockingOneWayExecutor (adapted) interface
        // --------------------------------------------------------------------
        template <typename F, typename... Ts>
        void post(F&& f, Ts&&... ts)
        {
            hpx::parallel::execution::post(executor_,
                throttling_wrapper<F>(*this, executor_, std::forward<F>(f)),
                std::forward<Ts>(ts)...);
        }

        // --------------------------------------------------------------------
        // BulkTwoWayExecutor interface
        template <typename F, typename S, typename... Ts>
        decltype(auto) bulk_async_execute(F&& f, S const& shape, Ts&&... ts)
        {
            return hpx::parallel::execution::bulk_async_execute(executor_,
                throttling_wrapper<F>(*this, executor_, std::forward<F>(f)),
                std::forward<Ts>(ts)...);
        }

        // --------------------------------------------------------------------
        template <typename F, typename S, typename Future, typename... Ts>
        decltype(auto) bulk_then_execute(
            F&& f, S const& shape, Future&& predecessor, Ts&&... ts)
        {
            return hpx::parallel::execution::bulk_then_execute(executor_,
                throttling_wrapper<F>(*this, executor_, std::forward<F>(f)),
                std::forward<Future>(predecessor), std::forward<Ts>(ts)...);
        }

        // --------------------------------------------------------------------
        // wait (suspend) until the number of tasks 'in flight' on this executor
        // drops to the lower threshold
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
        void count_up()
        {
            ++count_;
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
        mutable std::atomic<std::size_t> count_;
        mutable std::size_t lower_threshold_;
        mutable std::size_t upper_threshold_;
        bool block_;
    };
}}}    // namespace hpx::execution::experimental

namespace hpx { namespace parallel { namespace execution {

    // --------------------------------------------------------------------
    // simple forwarding implementations of executor traits
    // --------------------------------------------------------------------
    template <typename BaseExecutor>
    struct is_one_way_executor<
        hpx::execution::experimental::limiting_executor<BaseExecutor>>
      : is_one_way_executor<typename std::decay<BaseExecutor>::type>
    {
    };

    template <typename BaseExecutor>
    struct is_never_blocking_one_way_executor<
        hpx::execution::experimental::limiting_executor<BaseExecutor>>
      : is_never_blocking_one_way_executor<
            typename std::decay<BaseExecutor>::type>
    {
    };

    template <typename BaseExecutor>
    struct is_two_way_executor<
        hpx::execution::experimental::limiting_executor<BaseExecutor>>
      : is_two_way_executor<typename std::decay<BaseExecutor>::type>
    {
    };

    template <typename BaseExecutor>
    struct is_bulk_one_way_executor<
        hpx::execution::experimental::limiting_executor<BaseExecutor>>
      : is_bulk_one_way_executor<typename std::decay<BaseExecutor>::type>
    {
    };

    template <typename BaseExecutor>
    struct is_bulk_two_way_executor<
        hpx::execution::experimental::limiting_executor<BaseExecutor>>
      : is_bulk_two_way_executor<typename std::decay<BaseExecutor>::type>
    {
    };
}}}    // namespace hpx::parallel::execution

#include <hpx/config/warnings_suffix.hpp>
