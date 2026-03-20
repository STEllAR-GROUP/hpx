//  Copyright (c) 2026 Hartmut Kaiser
//  Copyright (c) 2026 Vansh Dobhal
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>

#include <cstddef>

#if defined(HPX_HAVE_MODULE_TRACY)
#include <hpx/modules/threading_base.hpp>
#include <hpx/modules/tracy.hpp>
#else
namespace hpx::threads {
    class thread_data;
}
#endif

namespace hpx::tracing {

    HPX_CXX_CORE_EXPORT struct region
    {
#if defined(HPX_HAVE_MODULE_TRACY)
        hpx::tracy::region impl;

        static hpx::tracy::region create_tracy_region(
            hpx::threads::thread_data* thrdptr, std::size_t num_thread) noexcept
        {
            char const* name = thrdptr->get_description().get_description();
            bool const enable_tracy =
                name != nullptr && !thrdptr->is_stackless();
            return hpx::tracy::region(
                name, num_thread, thrdptr->get_thread_phase(), enable_tracy);
        }
#endif

        region([[maybe_unused]] char const* name,
            [[maybe_unused]] std::size_t thread_num,
            [[maybe_unused]] std::size_t phase,
            [[maybe_unused]] bool enabled) noexcept
#if defined(HPX_HAVE_MODULE_TRACY)
          : impl(name, thread_num, phase, enabled)
#endif
        {
        }

        explicit region([[maybe_unused]] hpx::threads::thread_data* thrdptr,
            [[maybe_unused]] std::size_t num_thread) noexcept
#if defined(HPX_HAVE_MODULE_TRACY)
          : impl(create_tracy_region(thrdptr, num_thread))
#endif
        {
        }
    };

    HPX_CXX_CORE_EXPORT struct mark_event
    {
#if defined(HPX_HAVE_MODULE_TRACY)
        hpx::tracy::mark_event impl;
#endif

        explicit mark_event([[maybe_unused]] char const* name) noexcept
#if defined(HPX_HAVE_MODULE_TRACY)
          : impl(name)
#endif
        {
        }
    };

    HPX_CXX_CORE_EXPORT struct fiber_region
    {
#if defined(HPX_HAVE_MODULE_TRACY)
        hpx::tracy::fiber_region impl;

        static hpx::tracy::fiber_region create_tracy_fiber_region(
            hpx::threads::thread_data* thrdptr, std::size_t num_thread) noexcept
        {
            char const* name = thrdptr->get_description().get_description();
            bool const enable_tracy =
                name != nullptr && !thrdptr->is_stackless();
            char const* fiber_name =
                enable_tracy ? thrdptr->get_tracy_fiber_name() : nullptr;
            return hpx::tracy::fiber_region(
                fiber_name, name, num_thread, enable_tracy);
        }
#endif

        explicit fiber_region(
            [[maybe_unused]] hpx::threads::thread_data* thrdptr,
            [[maybe_unused]] std::size_t num_thread) noexcept
#if defined(HPX_HAVE_MODULE_TRACY)
          : impl(create_tracy_fiber_region(thrdptr, num_thread))
#endif
        {
        }
    };

}    // namespace hpx::tracing
