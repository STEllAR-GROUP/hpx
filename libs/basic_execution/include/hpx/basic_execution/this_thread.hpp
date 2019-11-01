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

#endif
