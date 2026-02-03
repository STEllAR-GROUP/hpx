//  Copyright (c) 2025-2026 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>

#if defined(HPX_HAVE_MODULE_TRACY)

#include <cstddef>
#include <cstdint>

///////////////////////////////////////////////////////////////////////////////
namespace hpx::tracy {

    HPX_CXX_EXPORT struct region_data
    {
        char const* name = nullptr;
        std::uint64_t data = 0;
        std::uint32_t color = 0;
        std::uint32_t phase = 0;
    };

    HPX_CXX_EXPORT HPX_CORE_EXPORT region_data start_region(
        char const*, std::size_t = 0, std::size_t = 0) noexcept;
    HPX_CXX_EXPORT HPX_CORE_EXPORT char const* rename_region(
        char const*) noexcept;
    HPX_CXX_EXPORT HPX_CORE_EXPORT region_data stop_region(
        region_data const& prev_region) noexcept;

    HPX_CXX_EXPORT struct region
    {
        explicit region(char const* name, std::size_t const thread_num,
            std::size_t phase) noexcept
          : surrounding_region(start_region(name, thread_num, phase))
        {
        }

        ~region() noexcept
        {
            stop_region(surrounding_region);
        }

    private:
        region_data surrounding_region;
    };

    HPX_CXX_EXPORT struct suspend_region
    {
        suspend_region() noexcept
          : suspended_region(stop_region({}))
        {
        }

        ~suspend_region() noexcept
        {
            if (suspended_region.name != nullptr)
            {
                start_region(suspended_region.name, suspended_region.color,
                    suspended_region.phase);
            }
        }

        region_data suspended_region;
    };

    HPX_CXX_EXPORT struct mark_event
    {
        explicit mark_event(char const* name) noexcept
          : previous_name(rename_region(name))
        {
        }

        ~mark_event()
        {
            rename_region(previous_name);
        }

    private:
        char const* previous_name;
    };
}    // namespace hpx::tracy

#endif
