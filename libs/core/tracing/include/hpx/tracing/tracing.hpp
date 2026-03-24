//  Copyright (c) 2026 Hartmut Kaiser
//  Copyright (c) 2026 Vansh Dobhal
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/modules/threading_base.hpp>

#include <cstddef>

#if defined(HPX_HAVE_MODULE_TRACY)
#include <hpx/modules/tracy.hpp>

namespace hpx::tracing {

    HPX_CXX_CORE_EXPORT struct HPX_CORE_EXPORT region
    {
        explicit region(hpx::threads::thread_data* thrdptr,
            std::size_t num_thread) noexcept;

        ~region();

    private:
        static hpx::tracy::region create_tracy_region(
            hpx::threads::thread_data* thrdptr,
            std::size_t num_thread) noexcept;

        hpx::tracy::region impl;
    };

    HPX_CXX_CORE_EXPORT struct HPX_CORE_EXPORT mark_event
    {
        explicit mark_event(char const* name) noexcept;
        ~mark_event();

    private:
        hpx::tracy::mark_event impl;
    };

    HPX_CXX_CORE_EXPORT struct HPX_CORE_EXPORT fiber_region
    {
        explicit fiber_region(hpx::threads::thread_data* thrdptr,
            std::size_t num_thread) noexcept;

        ~fiber_region();

    private:
        static hpx::tracy::fiber_region create_tracy_fiber_region(
            hpx::threads::thread_data* thrdptr,
            std::size_t num_thread) noexcept;

        hpx::tracy::fiber_region impl;
    };

}    // namespace hpx::tracing

#else

namespace hpx::tracing {

    HPX_CXX_CORE_EXPORT struct [[maybe_unused]] region
    {
        constexpr explicit region(
            hpx::threads::thread_data*, std::size_t) noexcept
        {
        }
    };

    HPX_CXX_CORE_EXPORT struct [[maybe_unused]] mark_event
    {
        constexpr explicit mark_event(char const*) noexcept {}
    };

    HPX_CXX_CORE_EXPORT struct [[maybe_unused]] fiber_region
    {
        constexpr explicit fiber_region(
            hpx::threads::thread_data*, std::size_t) noexcept
        {
        }
    };

}    // namespace hpx::tracing

#endif
