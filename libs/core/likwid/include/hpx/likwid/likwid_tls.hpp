//  Copyright (c) 2022 Srinivas Yadav
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>

#if defined(HPX_HAVE_MODULE_LIKWID)

///////////////////////////////////////////////////////////////////////////////
namespace hpx::likwid {

    HPX_EXPORT_CORE char const* start_region(char const*) noexcept;
    HPX_EXPORT_CORE char const* stop_region() noexcept;

    struct suspend_region
    {
        suspend_region() noexcept
          : region(stop_region())
        {
        }
        ~suspend_region() noexcept
        {
            if (region != nullptr)
            {
                start_region(region);
            }
        }

        char const* region = nullptr;
    };
}    // namespace hpx::likwid

#endif
