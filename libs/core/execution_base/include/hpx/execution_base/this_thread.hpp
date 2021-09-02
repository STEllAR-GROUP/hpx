//  Copyright (c) 2019 Thomas Heller
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/local/config.hpp>
#include <hpx/execution_base/agent_base.hpp>
#include <hpx/execution_base/agent_ref.hpp>
#include <hpx/execution_base/detail/spinlock_deadlock_detection.hpp>
#include <hpx/timing/high_resolution_timer.hpp>
#include <hpx/timing/steady_clock.hpp>

#ifdef HPX_HAVE_SPINLOCK_DEADLOCK_DETECTION
#include <hpx/errors/throw_exception.hpp>
#endif

#include <chrono>
#include <cstddef>
#include <cstdint>

namespace hpx { namespace execution_base {
    namespace detail {
        HPX_LOCAL_EXPORT agent_base& get_default_agent();
    }

    ///////////////////////////////////////////////////////////////////////////
    namespace this_thread {
        namespace detail {

            struct agent_storage;
            HPX_LOCAL_EXPORT agent_storage* get_agent_storage();
        }    // namespace detail

        struct HPX_LOCAL_EXPORT reset_agent
        {
            reset_agent(detail::agent_storage*, agent_base& impl);
            reset_agent(agent_base& impl);
            ~reset_agent();

            detail::agent_storage* storage_;
            agent_base* old_;
        };

        HPX_LOCAL_EXPORT hpx::execution_base::agent_ref agent();

        HPX_LOCAL_EXPORT void yield(
            char const* desc = "hpx::execution_base::this_thread::yield");
        HPX_LOCAL_EXPORT void yield_k(std::size_t k,
            char const* desc = "hpx::execution_base::this_thread::yield_k");
        HPX_LOCAL_EXPORT void suspend(
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
}}       // namespace hpx::execution_base

namespace hpx { namespace util {
    namespace detail {
        inline void yield_k(std::size_t k, const char* thread_name)
        {
#ifdef HPX_HAVE_SPINLOCK_DEADLOCK_DETECTION
            if (k > 32 && get_spinlock_break_on_deadlock_enabled() &&
                k > get_spinlock_deadlock_detection_limit())
            {
                HPX_THROW_EXCEPTION(
                    deadlock, thread_name, "possible deadlock detected");
            }
#endif
            hpx::execution_base::this_thread::yield_k(k, thread_name);
        }
    }    // namespace detail

    template <typename Predicate>
    void yield_while(Predicate&& predicate, const char* thread_name = nullptr,
        bool allow_timed_suspension = true)
    {
        if (allow_timed_suspension)
        {
            for (std::size_t k = 0; predicate(); ++k)
            {
                detail::yield_k(k, thread_name);
            }
        }
        else
        {
            for (std::size_t k = 0; predicate(); ++k)
            {
                detail::yield_k(k % 16, thread_name);
            }
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
            std::size_t required_count, const char* thread_name = nullptr,
            bool allow_timed_suspension = true)
        {
            std::size_t count = 0;
            if (allow_timed_suspension)
            {
                for (std::size_t k = 0;; ++k)
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
                        detail::yield_k(k, thread_name);
                    }
                }
            }
            else
            {
                for (std::size_t k = 0;; ++k)
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
                        detail::yield_k(k % 16, thread_name);
                    }
                }
            }
        }

        // yield_while_count_timeout is similar to yield_while_count, with the
        // addition of a timeout parameter. If the timeout is exceeded, waiting
        // is stopped and the function returns false. If the predicate is
        // successfully waited for the function returns true.
        template <typename Predicate>
        HPX_NODISCARD bool yield_while_count_timeout(Predicate&& predicate,
            std::size_t required_count, std::chrono::duration<double> timeout,
            const char* thread_name = nullptr,
            bool allow_timed_suspension = true)
        {
            // Seconds represented using a double
            using duration_type = std::chrono::duration<double>;

            bool use_timeout = timeout >= duration_type(0.0);

            std::size_t count = 0;
            hpx::chrono::high_resolution_timer t;

            if (allow_timed_suspension)
            {
                for (std::size_t k = 0;; ++k)
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
                        detail::yield_k(k, thread_name);
                    }
                }
            }
            else
            {
                for (std::size_t k = 0;; ++k)
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
                        detail::yield_k(k % 16, thread_name);
                    }
                }
            }
        }
    }    // namespace detail
}}       // namespace hpx::util
