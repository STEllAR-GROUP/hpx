//  Copyright (c) 2019 Thomas Heller
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_BASIC_EXECUTION_AGENT_REF_HPP
#define HPX_BASIC_EXECUTION_AGENT_REF_HPP

#include <hpx/config.hpp>
#include <hpx/timing/steady_clock.hpp>

#include <chrono>
#include <iosfwd>

namespace hpx { namespace basic_execution {

    struct agent_base;

    class HPX_EXPORT agent_ref
    {
    public:
        constexpr agent_ref() noexcept
          : impl_(nullptr)
        {
        }
        constexpr agent_ref(agent_base* impl) noexcept
          : impl_(impl)
        {
        }

        constexpr agent_ref(agent_ref const&) noexcept = default;
        HPX_CXX14_CONSTEXPR agent_ref& operator=(
            agent_ref const&) noexcept = default;

        constexpr agent_ref(agent_ref&&) noexcept = default;
        HPX_CXX14_CONSTEXPR agent_ref& operator=(
            agent_ref&&) noexcept = default;

        constexpr explicit operator bool() noexcept
        {
            return impl_ != nullptr;
        }

        void reset(agent_base* impl = nullptr)
        {
            impl_ = impl;
        }

        void yield(char const* desc = "hpx::basic_execution::agent_ref::yield");
        void yield_k(std::size_t k,
            char const* desc = "hpx::basic_execution::agent_ref::yield_k");
        void suspend(
            char const* desc = "hpx::basic_execution::agent_ref::suspend");
        void resume(
            char const* desc = "hpx::basic_execution::agent_ref::resume");
        void abort(char const* desc = "hpx::basic_execution::agent_ref::abort");

        template <typename Rep, typename Period>
        void sleep_for(std::chrono::duration<Rep, Period> const& sleep_duration,
            char const* desc = "hpx::basic_execution::agent_ref::sleep_for")
        {
            sleep_for(hpx::util::steady_duration{sleep_duration}, desc);
        }

        template <typename Clock, typename Duration>
        void sleep_until(
            std::chrono::time_point<Clock, Duration> const& sleep_time,
            char const* desc = "hpx::basic_execution::agent_ref::sleep_until")
        {
            sleep_until(hpx::util::steady_time_point{sleep_time}, desc);
        }

        // TODO:
        // affinity
        // thread_num
        // executor

    private:
        agent_base* impl_;

        void sleep_for(
            hpx::util::steady_duration const& sleep_duration, char const* desc);
        void sleep_until(
            hpx::util::steady_time_point const& sleep_time, char const* desc);

        friend constexpr bool operator==(
            agent_ref const& lhs, agent_ref const& rhs)
        {
            return lhs.impl_ == rhs.impl_;
        }

        friend constexpr bool operator!=(
            agent_ref const& lhs, agent_ref const& rhs)
        {
            return lhs.impl_ != rhs.impl_;
        }

        friend std::ostream& operator<<(std::ostream&, agent_ref const&);
    };
}}    // namespace hpx::basic_execution

#endif
