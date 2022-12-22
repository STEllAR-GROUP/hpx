//  Copyright (c) 2016-2022 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/synchronization/detail/condition_variable.hpp>
#include <hpx/synchronization/spinlock.hpp>

#include <cstdint>
#include <mutex>
#include <utility>

#if defined(HPX_MSVC_WARNING_PRAGMA)
#pragma warning(push)
#pragma warning(disable : 4251)
#endif

////////////////////////////////////////////////////////////////////////////////
namespace hpx::lcos::local::detail {

    class sliding_semaphore
    {
    private:
        using mutex_type = hpx::spinlock;

    public:
        HPX_CORE_EXPORT sliding_semaphore(
            std::int64_t max_difference, std::int64_t lower_limit) noexcept;
        HPX_CORE_EXPORT ~sliding_semaphore();

        HPX_CORE_EXPORT void set_max_difference(std::unique_lock<mutex_type>& l,
            std::int64_t max_difference, std::int64_t lower_limit) noexcept;

        HPX_CORE_EXPORT void wait(
            std::unique_lock<mutex_type>& l, std::int64_t upper_limit);

        HPX_CORE_EXPORT bool try_wait(
            std::unique_lock<mutex_type>& l, std::int64_t upper_limit);

        HPX_CORE_EXPORT void signal(
            std::unique_lock<mutex_type> l, std::int64_t lower_limit);

        HPX_CORE_EXPORT std::int64_t signal_all(std::unique_lock<mutex_type> l);

    private:
        std::int64_t max_difference_;
        std::int64_t lower_limit_;
        local::detail::condition_variable cond_;
    };
}    // namespace hpx::lcos::local::detail

#if defined(HPX_MSVC_WARNING_PRAGMA)
#pragma warning(pop)
#endif
