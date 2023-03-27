//  Copyright (c) 2019 Thomas Heller
//  Copyright (c) 2022 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/execution_base/agent_base.hpp>
#include <hpx/execution_base/agent_ref.hpp>
#include <hpx/execution_base/sender.hpp>
#include <hpx/functional/detail/tag_fallback_invoke.hpp>
#include <hpx/modules/type_support.hpp>
#include <hpx/timing/high_resolution_timer.hpp>
#include <hpx/timing/steady_clock.hpp>

#ifdef HPX_HAVE_SPINLOCK_DEADLOCK_DETECTION
#include <hpx/errors/throw_exception.hpp>
#include <hpx/execution_base/detail/spinlock_deadlock_detection.hpp>
#endif

#include <chrono>
#include <cstddef>
#include <cstdint>

namespace hpx::execution_base {

    namespace detail {

        HPX_CORE_EXPORT agent_base& get_default_agent();
    }

    ///////////////////////////////////////////////////////////////////////////
    namespace this_thread {

        namespace detail {

            struct agent_storage;
            HPX_CORE_EXPORT agent_storage* get_agent_storage();
        }    // namespace detail

        struct HPX_CORE_EXPORT reset_agent
        {
            explicit reset_agent(agent_base& impl);
            reset_agent(detail::agent_storage*, agent_base& impl);

            reset_agent(reset_agent const&) = delete;
            reset_agent(reset_agent&&) = delete;
            reset_agent& operator=(reset_agent const&) = delete;
            reset_agent& operator=(reset_agent&&) = delete;

            ~reset_agent();

            detail::agent_storage* storage_;
            agent_base* old_;
        };

        HPX_CORE_EXPORT hpx::execution_base::agent_ref agent();

        HPX_CORE_EXPORT void yield(
            char const* desc = "hpx::execution_base::this_thread::yield");
        HPX_CORE_EXPORT void yield_k(std::size_t k,
            char const* desc = "hpx::execution_base::this_thread::yield_k");
        HPX_CORE_EXPORT void suspend(
            char const* desc = "hpx::execution_base::this_thread::suspend");

        template <typename Rep, typename Period>
        void sleep_for(std::chrono::duration<Rep, Period> const& sleep_duration,
            char const* desc = "hpx::execution_base::this_thread::sleep_for")
        {
            agent().sleep_for(sleep_duration, desc);
        }

        template <class Clock, class Duration>
        void sleep_until(
            std::chrono::time_point<Clock, Duration> const& sleep_time,
            char const* desc = "hpx::execution_base::this_thread::sleep_for")
        {
            agent().sleep_until(sleep_time, desc);
        }
    }    // namespace this_thread
}    // namespace hpx::execution_base

namespace hpx::this_thread::experimental {

    // [exec.sched_queries.execute_may_block_caller]
    //
    // 1. `this_thread::execute_may_block_caller` is used to ask a scheduler s
    // whether a call `execution::execute(s, f)` with any invocable f may block
    // the thread where such a call occurs.
    //
    // 2. The name `this_thread::execute_may_block_caller` denotes a
    // customization point object. For some subexpression s, let S be
    // decltype((s)). If S does not satisfy `execution::scheduler`,
    // `this_thread::execute_may_block_caller` is ill-formed. Otherwise,
    // `this_thread::execute_may_block_caller(s)` is expression equivalent to:
    //
    //      1. `tag_invoke(this_thread::execute_may_block_caller, as_const(s))`,
    //          if this expression is well formed.
    //
    //          -- Mandates: The tag_invoke expression above is not
    //                       potentially throwing and its type is bool.
    //
    //      2. Otherwise, true.
    //
    // 3. If `this_thread::execute_may_block_caller(s)` for some scheduler s
    // returns false, no execution::execute(s, f) call with some invocable f
    // shall block the calling thread.
    namespace detail {

        // apply this meta function to all tag_invoke variations
        struct is_scheduler
        {
            template <typename EnableTag, typename... T>
            using apply = hpx::execution::experimental::is_scheduler<T...>;
        };
    }    // namespace detail

    HPX_HOST_DEVICE_INLINE_CONSTEXPR_VARIABLE struct execute_may_block_caller_t
        final
      : hpx::functional::detail::tag_fallback_noexcept<
            execute_may_block_caller_t, detail::is_scheduler>
    {
    private:
        template <typename T>
        friend constexpr HPX_FORCEINLINE bool tag_fallback_invoke(
            execute_may_block_caller_t, T const&) noexcept
        {
            return true;
        }
    } execute_may_block_caller{};
}    // namespace hpx::this_thread::experimental

namespace hpx::util {

    namespace detail {

        inline void yield_k(std::size_t k, char const* thread_name)
        {
#ifdef HPX_HAVE_SPINLOCK_DEADLOCK_DETECTION
            if (k > 32 && get_spinlock_break_on_deadlock_enabled() &&
                k > get_spinlock_deadlock_detection_limit())
            {
                HPX_THROW_EXCEPTION(hpx::error::deadlock, thread_name,
                    "possible deadlock detected");
            }
#endif
            hpx::execution_base::this_thread::yield_k(k, thread_name);
        }
    }    // namespace detail

    template <bool AllowTimedSuspension, typename Predicate>
    void yield_while(Predicate&& predicate, char const* thread_name = nullptr)
    {
        for (std::size_t k = 0; predicate(); ++k)
        {
            if constexpr (AllowTimedSuspension)
            {
                detail::yield_k(k, thread_name);
            }
            else
            {
                detail::yield_k(k % 16, thread_name);
            }
        }
    }

    template <typename Predicate>
    void yield_while(Predicate&& predicate, char const* thread_name = nullptr,
        bool allow_timed_suspension = true)
    {
        if (allow_timed_suspension)
        {
            yield_while<true>(HPX_FORWARD(Predicate, predicate), thread_name);
        }
        else
        {
            yield_while<false>(HPX_FORWARD(Predicate, predicate), thread_name);
        }
    }

    namespace detail {

        // yield_while_count yields until the predicate returns true
        // required_count times consecutively. This function is used in cases
        // where there is a small false positive rate and repeatedly calling the
        // predicate reduces the rate of false positives overall.
        //
        // Note: This is mostly a hack used to work around the raciness of
        // termination detection for thread pools and the runtime and can be
        // replaced if and when a better solution appears.
        template <typename Predicate>
        void yield_while_count(Predicate&& predicate,
            std::size_t required_count, char const* thread_name = nullptr,
            bool allow_timed_suspension = true)
        {
            std::size_t count = 0;
            for (std::size_t k = 0; /**/; ++k)
            {
                if (!predicate())
                {
                    if (++count > required_count)
                    {
                        return;
                    }
                }
                else
                {
                    count = 0;
                    detail::yield_k(
                        allow_timed_suspension ? k : k % 16, thread_name);
                }
            }
        }

        // yield_while_count_timeout is similar to yield_while_count, with the
        // addition of a timeout parameter. If the timeout is exceeded, waiting
        // is stopped and the function returns false. If the predicate is
        // successfully waited for the function returns true.
        template <typename Predicate>
        bool yield_while_count_timeout(Predicate&& predicate,
            std::size_t required_count, std::chrono::duration<double> timeout,
            char const* thread_name = nullptr,
            bool allow_timed_suspension = true)
        {
            // Seconds represented using a double
            using duration_type = std::chrono::duration<double>;

            // Initialize timer only if needed
            bool const use_timeout = timeout >= duration_type(0.0);
            hpx::chrono::high_resolution_timer t(
                hpx::chrono::high_resolution_timer::init::no_init);

            if (use_timeout)
            {
                t.restart();
            }

            std::size_t count = 0;
            for (std::size_t k = 0; /**/; ++k)
            {
                if (use_timeout && duration_type(t.elapsed()) > timeout)
                {
                    return false;
                }

                if (!predicate())
                {
                    if (++count > required_count)
                    {
                        return true;
                    }
                }
                else
                {
                    count = 0;
                    detail::yield_k(
                        allow_timed_suspension ? k : k % 16, thread_name);
                }
            }
        }
    }    // namespace detail
}    // namespace hpx::util
