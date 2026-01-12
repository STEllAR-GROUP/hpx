//  Copyright (c) 2022 Srinivas Yadav
//  Copyright (c) 2025 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>

#if defined(HPX_HAVE_MODULE_LIKWID)

///////////////////////////////////////////////////////////////////////////////
namespace hpx::likwid {

    HPX_CXX_EXPORT HPX_CORE_EXPORT char const* start_region(
        char const*) noexcept;
    HPX_CXX_EXPORT HPX_CORE_EXPORT char const* stop_region() noexcept;

    HPX_CXX_EXPORT struct region
    {
        region(char const* name) noexcept
          : surrounding_region(start_region(name))
        {
        }
        ~region() noexcept
        {
            if (surrounding_region != nullptr)
            {
                stop_region();
            }
        }

        char const* surrounding_region;
    };

    HPX_CXX_EXPORT struct suspend_region
    {
        suspend_region() noexcept
          : suspended_region(stop_region())
        {
        }
        ~suspend_region() noexcept
        {
            if (suspended_region != nullptr)
            {
                start_region(suspended_region);
            }
        }

        char const* suspended_region;
    };
}    // namespace hpx::likwid

#endif
