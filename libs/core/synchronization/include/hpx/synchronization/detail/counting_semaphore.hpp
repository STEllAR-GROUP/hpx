//  Copyright (c) 2007-2020 Hartmut Kaiser
//  Copyright (c)      2011 Bryce Lelbach
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/local/config.hpp>
#include <hpx/synchronization/detail/condition_variable.hpp>
#include <hpx/synchronization/spinlock.hpp>

#include <cstddef>
#include <cstdint>
#include <mutex>

#if defined(HPX_MSVC_WARNING_PRAGMA)
#pragma warning(push)
#pragma warning(disable : 4251)
#endif

////////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace lcos { namespace local { namespace detail {

    class counting_semaphore
    {
    private:
        typedef lcos::local::spinlock mutex_type;

    public:
        HPX_LOCAL_EXPORT counting_semaphore(std::ptrdiff_t value = 0);
        HPX_LOCAL_EXPORT ~counting_semaphore();

        HPX_LOCAL_EXPORT void wait(
            std::unique_lock<mutex_type>& l, std::ptrdiff_t count);

        HPX_LOCAL_EXPORT bool wait_until(std::unique_lock<mutex_type>& l,
            hpx::chrono::steady_time_point const& abs_time,
            std::ptrdiff_t count);

        HPX_LOCAL_EXPORT bool try_wait(
            std::unique_lock<mutex_type>& l, std::ptrdiff_t count = 1);

        HPX_LOCAL_EXPORT bool try_acquire(std::unique_lock<mutex_type>& l);

        HPX_LOCAL_EXPORT void signal(
            std::unique_lock<mutex_type> l, std::ptrdiff_t count);

        HPX_LOCAL_EXPORT std::ptrdiff_t signal_all(
            std::unique_lock<mutex_type> l);

    private:
        std::ptrdiff_t value_;
        local::detail::condition_variable cond_;
    };
}}}}    // namespace hpx::lcos::local::detail

#if defined(HPX_MSVC_WARNING_PRAGMA)
#pragma warning(pop)
#endif
