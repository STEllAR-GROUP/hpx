//  Copyright (c) 2026 Hartmut Kaiser
//  Copyright (c) 2026 Vansh Dobhal
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/tracing/tracing.hpp>

#include <cstddef>

#if defined(HPX_HAVE_MODULE_TRACY)

namespace hpx::tracing {

    ////////////////////////////////////////////////////////////////////////////
    // region

    hpx::tracy::region region::create_tracy_region(
        region_init_data const& data, std::size_t const num_thread) noexcept
    {
        bool const enabled = data.name != nullptr && !data.is_stackless;
        return hpx::tracy::region(
            data.name, num_thread, data.thread_phase, enabled);
    }

    region::region(
        region_init_data const& data, std::size_t const num_thread) noexcept
      : impl(create_tracy_region(data, num_thread))
    {
    }

    region::~region() = default;

    ////////////////////////////////////////////////////////////////////////////
    // mark_event

    mark_event::mark_event(char const* name) noexcept
      : impl(name)
    {
    }

    mark_event::~mark_event() = default;

    ////////////////////////////////////////////////////////////////////////////
    // fiber_region

    hpx::tracy::fiber_region fiber_region::create_tracy_fiber_region(
        fiber_region_init_data const& data,
        std::size_t const num_thread) noexcept
    {
        bool const enabled = data.name != nullptr && !data.is_stackless;
        char const* fiber_name = enabled ? data.fiber_name : nullptr;

        // Use num_thread as color seed so each worker thread gets a distinct
        // color on the fiber track in Tracy.
        auto const color =
            static_cast<std::size_t>(num_thread + 1) * 0x9e3779b9;
        return hpx::tracy::fiber_region(fiber_name, data.name, color, enabled);
    }

    fiber_region::fiber_region(fiber_region_init_data const& data,
        std::size_t const num_thread) noexcept
      : impl(create_tracy_fiber_region(data, num_thread))
    {
    }

    fiber_region::~fiber_region() = default;

    ////////////////////////////////////////////////////////////////////////////
    // fiber_suspend_region

    fiber_suspend_region::fiber_suspend_region(char const* desc) noexcept
      : impl(desc)
    {
    }

    fiber_suspend_region::~fiber_suspend_region() = default;

    ////////////////////////////////////////////////////////////////////////////
    // set_thread_name

    set_thread_name::set_thread_name(char const* name) noexcept
    {
        hpx::tracy::set_thread_name(name);
    }

    set_thread_name::~set_thread_name() = default;

}    // namespace hpx::tracing

#endif
