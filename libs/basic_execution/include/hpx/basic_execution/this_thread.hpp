//  Copyright (c) 2019 Thomas Heller
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_BASIC_EXECUTION_THIS_THREAD_HPP
#define HPX_BASIC_EXECUTION_THIS_THREAD_HPP

#include <hpx/config.hpp>
#include <hpx/basic_execution/agent_base.hpp>
#include <hpx/basic_execution/agent_ref.hpp>
#include <hpx/timing/steady_clock.hpp>

#ifdef HPX_HAVE_SPINLOCK_DEADLOCK_DETECTION
#include <hpx/errors/throw_exception.hpp>
#endif

#include <chrono>
#include <cstddef>
#include <cstdint>

namespace hpx { namespace basic_execution { namespace this_thread {

    namespace detail {

        struct agent_storage;
        HPX_EXPORT agent_storage* get_agent_storage();
    }    // namespace detail

    struct HPX_EXPORT reset_agent
    {
        reset_agent(detail::agent_storage*, agent_base& impl);
        reset_agent(agent_base& impl);
        ~reset_agent();

        detail::agent_storage* storage_;
        agent_base* old_;
    };

    HPX_EXPORT hpx::basic_execution::agent_ref agent();

    HPX_EXPORT void yield(
        char const* desc = "hpx::basic_execution::this_thread::yield");
    HPX_EXPORT void yield_k(std::size_t k,
        char const* desc = "hpx::basic_execution::this_thread::yield_k");
    HPX_EXPORT void suspend(
        char const* desc = "hpx::basic_execution::this_thread::suspend");

    template <typename Rep, typename Period>
    void sleep_for(std::chrono::duration<Rep, Period> const& sleep_duration,
        char const* desc = "hpx::basic_execution::this_thread::sleep_for")
    {
        agent().sleep_for(sleep_duration, desc);
    }

    template <class Clock, class Duration>
    void sleep_until(std::chrono::time_point<Clock, Duration> const& sleep_time,
        char const* desc = "hpx::basic_execution::this_thread::sleep_for")
    {
        agent().sleep_until(sleep_time, desc);
    }
}}}    // namespace hpx::basic_execution::this_thread

namespace hpx { namespace util {
    namespace detail {
#ifdef HPX_HAVE_SPINLOCK_DEADLOCK_DETECTION
        HPX_API_EXPORT extern bool spinlock_break_on_deadlock;
        HPX_API_EXPORT extern std::size_t spinlock_deadlock_detection_limit;
#endif

        inline void yield_k(std::size_t k, const char* thread_name)
        {
#ifdef HPX_HAVE_SPINLOCK_DEADLOCK_DETECTION
            if (k > 32 && spinlock_break_on_deadlock &&
                k > spinlock_deadlock_detection_limit)
            {
                HPX_THROW_EXCEPTION(
                    deadlock, thread_name, "possible deadlock detected");
            }
#endif
            hpx::basic_execution::this_thread::yield_k(k, thread_name);
        }
    }    // namespace detail

    template <typename Predicate>
    inline void yield_while(Predicate&& predicate,
        const char* thread_name = nullptr, bool allow_timed_suspension = true)
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
                detail::yield_k(k % 32, thread_name);
            }
        }
    }
}}    // namespace hpx::util

#endif
