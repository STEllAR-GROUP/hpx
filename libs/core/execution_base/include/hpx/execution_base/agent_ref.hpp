//  Copyright (c) 2019 Thomas Heller
//  Copyright (c) 2023 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/timing/steady_clock.hpp>

#include <chrono>
#include <cstddef>
#include <iosfwd>

namespace hpx::execution_base {

    struct agent_base;

    class HPX_CORE_EXPORT agent_ref
    {
    public:
        agent_ref() = default;
        ~agent_ref() = default;

        explicit constexpr agent_ref(agent_base* impl) noexcept
          : impl_(impl)
        {
        }

        agent_ref(agent_ref const&) = default;
        agent_ref& operator=(agent_ref const&) = default;
        agent_ref(agent_ref&&) = default;
        agent_ref& operator=(agent_ref&&) = default;

        explicit constexpr operator bool() const noexcept
        {
            return impl_ != nullptr;
        }

        void reset(agent_base* impl = nullptr) noexcept
        {
            impl_ = impl;
        }

        void yield(
            char const* desc = "hpx::execution_base::agent_ref::yield") const;
        void yield_k(std::size_t k,
            char const* desc = "hpx::execution_base::agent_ref::yield_k") const;
        void suspend(
            char const* desc = "hpx::execution_base::agent_ref::suspend") const;
        void resume(
            char const* desc = "hpx::execution_base::agent_ref::resume") const;
        void abort(
            char const* desc = "hpx::execution_base::agent_ref::abort") const;

        template <typename Rep, typename Period>
        void sleep_for(std::chrono::duration<Rep, Period> const& sleep_duration,
            char const* desc = "hpx::execution_base::agent_ref::sleep_for")
        {
            sleep_for(hpx::chrono::steady_duration{sleep_duration}, desc);
        }

        template <typename Clock, typename Duration>
        void sleep_until(
            std::chrono::time_point<Clock, Duration> const& sleep_time,
            char const* desc = "hpx::execution_base::agent_ref::sleep_until")
        {
            sleep_until(hpx::chrono::steady_time_point{sleep_time}, desc);
        }

        [[nodiscard]] agent_base& ref() const noexcept
        {
            return *impl_;
        }

        // TODO:
        // affinity
        // thread_num
        // executor

    private:
        agent_base* impl_ = nullptr;

        void sleep_for(hpx::chrono::steady_duration const& sleep_duration,
            char const* desc) const;
        void sleep_until(hpx::chrono::steady_time_point const& sleep_time,
            char const* desc) const;

        friend constexpr bool operator==(
            agent_ref const& lhs, agent_ref const& rhs) noexcept
        {
            return lhs.impl_ == rhs.impl_;
        }

        friend constexpr bool operator!=(
            agent_ref const& lhs, agent_ref const& rhs) noexcept
        {
            return lhs.impl_ != rhs.impl_;
        }

        HPX_CORE_EXPORT friend std::ostream& operator<<(
            std::ostream&, agent_ref const&);
    };
}    // namespace hpx::execution_base
