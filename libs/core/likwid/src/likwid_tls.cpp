//  Copyright (c) 2022 Srinivas Yadav
//  Copyright (c) 2025 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>

#if defined(HPX_HAVE_MODULE_LIKWID)
#include <hpx/likwid/likwid_tls.hpp>

#include <likwid.h>

///////////////////////////////////////////////////////////////////////////////
namespace hpx::likwid {

    char const*& current_region() noexcept
    {
        thread_local char const* region = nullptr;
        return region;
    }

    char const* start_region(char const* new_region) noexcept
    {
        char const* current = current_region();

        likwid_markerStartRegion(new_region);
        current_region() = new_region;

        return current;
    }

    char const* stop_region(char const* curr_region) noexcept
    {
        char const* current = current_region();

        current_region() = curr_region;
        if (current != nullptr)
        {
            likwid_markerStopRegion(current);
        }

        return current;
    }
}    // namespace hpx::likwid

#endif
