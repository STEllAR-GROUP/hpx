//  Copyright (c) 2026 Hartmut Kaiser
//  Copyright (c) 2026 Vansh Dobhal
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/tracing/tracing.hpp>

#if defined(HPX_HAVE_MODULE_TRACY)

namespace hpx::tracing {

    // region

    hpx::tracy::region region::create_tracy_region(
        hpx::threads::thread_data* thrdptr, std::size_t num_thread) noexcept
    {
        char const* name = thrdptr->get_description().get_description();
        bool const enabled = name != nullptr && !thrdptr->is_stackless();
        return hpx::tracy::region(
            name, num_thread, thrdptr->get_thread_phase(), enabled);
    }

    region::region(char const* name, std::size_t num_thread, std::size_t phase,
        bool enabled) noexcept
      : impl(name, num_thread, phase, enabled)
    {
    }

    region::region(
        hpx::threads::thread_data* thrdptr, std::size_t num_thread) noexcept
      : impl(create_tracy_region(thrdptr, num_thread))
    {
    }

    region::~region() = default;

    // mark_event

    mark_event::mark_event(char const* name) noexcept
      : impl(name)
    {
    }

    mark_event::~mark_event() = default;

    // fiber_region

    hpx::tracy::fiber_region fiber_region::create_tracy_fiber_region(
        hpx::threads::thread_data* thrdptr,
        [[maybe_unused]] std::size_t num_thread) noexcept
    {
        char const* name = thrdptr->get_description().get_description();
        bool const enabled = name != nullptr && !thrdptr->is_stackless();
        char const* fiber_name =
            enabled ? thrdptr->get_tracy_fiber_name() : nullptr;
        return hpx::tracy::fiber_region(fiber_name, name, 0, enabled);
    }

    fiber_region::fiber_region(
        hpx::threads::thread_data* thrdptr, std::size_t num_thread) noexcept
      : impl(create_tracy_fiber_region(thrdptr, num_thread))
    {
    }

    fiber_region::~fiber_region() = default;

}    // namespace hpx::tracing

#endif
