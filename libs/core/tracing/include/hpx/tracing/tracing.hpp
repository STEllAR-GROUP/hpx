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

    struct HPX_CORE_EXPORT region
    {
        region(char const* name, std::size_t num_thread, std::size_t phase,
            bool enabled) noexcept;

        explicit region(hpx::threads::thread_data* thrdptr,
            std::size_t num_thread) noexcept;

        ~region();

    private:
        static hpx::tracy::region create_tracy_region(
            hpx::threads::thread_data* thrdptr,
            std::size_t num_thread) noexcept;

        hpx::tracy::region impl;
    };

    struct HPX_CORE_EXPORT mark_event
    {
        explicit mark_event(char const* name) noexcept;
        ~mark_event();

    private:
        hpx::tracy::mark_event impl;
    };

    struct HPX_CORE_EXPORT fiber_region
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

    struct region
    {
        constexpr region([[maybe_unused]] char const* name,
            [[maybe_unused]] std::size_t num_thread,
            [[maybe_unused]] std::size_t phase,
            [[maybe_unused]] bool enabled) noexcept
        {
        }

        constexpr explicit region(
            [[maybe_unused]] hpx::threads::thread_data* thrdptr,
            [[maybe_unused]] std::size_t num_thread) noexcept
        {
        }
    };

    struct mark_event
    {
        constexpr explicit mark_event(
            [[maybe_unused]] char const* name) noexcept
        {
        }
    };

    struct fiber_region
    {
        constexpr explicit fiber_region(
            [[maybe_unused]] hpx::threads::thread_data* thrdptr,
            [[maybe_unused]] std::size_t num_thread) noexcept
        {
        }
    };

}    // namespace hpx::tracing

#endif
