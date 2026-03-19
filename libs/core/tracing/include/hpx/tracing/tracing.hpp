//  Copyright (c) 2026 Vansh Dobhal
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>

#include <cstddef>

#if defined(HPX_HAVE_MODULE_TRACY)
#include <hpx/tracy/tracy_tls.hpp>
#endif

namespace hpx::tracing {

    HPX_CXX_CORE_EXPORT struct region
    {
#if defined(HPX_HAVE_MODULE_TRACY)
        hpx::tracy::region impl;
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

}    // namespace hpx::tracing
