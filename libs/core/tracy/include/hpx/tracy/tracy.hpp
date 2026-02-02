//  Copyright (c) 2026 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>

#if defined(HPX_HAVE_MODULE_TRACY)

#include <string>

namespace hpx::tracy {

    HPX_CXX_EXPORT HPX_CORE_EXPORT void set_thread_name(
        char const* name) noexcept;

    HPX_CXX_EXPORT HPX_CORE_EXPORT void enter_fiber(char const* name) noexcept;
    HPX_CXX_EXPORT HPX_CORE_EXPORT void leave_fiber() noexcept;

    HPX_CXX_EXPORT struct fiber_region
    {
        explicit fiber_region(char const* name) noexcept
        {
            enter_fiber(name);
        }

        ~fiber_region() noexcept
        {
            leave_fiber();
        }
    };

    HPX_CXX_EXPORT HPX_CORE_EXPORT void create_counter(
        std::string const& name) noexcept;

    HPX_CXX_EXPORT HPX_CORE_EXPORT void sample_value(
        std::string const& name, double value) noexcept;

}    // namespace hpx::tracy

#endif
