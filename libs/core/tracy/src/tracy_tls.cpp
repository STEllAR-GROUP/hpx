//  Copyright (c) 2025-2026 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>

#if defined(HPX_HAVE_MODULE_TRACY)
#include <hpx/tracy/tracy_tls.hpp>

#include <cstddef>
#include <cstdint>
#include <cstring>

#include <tracy/TracyC.h>

///////////////////////////////////////////////////////////////////////////////
namespace hpx::tracy {

    namespace {

        // Store currently active Tracy zone
        region_data& current_region() noexcept
        {
            thread_local region_data region;
            return region;
        }

        // This union is used to hide the Tracy context type from the user.
        union tracy_context
        {
            static_assert(sizeof(TracyCZoneCtx) == sizeof(std::uint64_t));

            TracyCZoneCtx context;
            std::uint64_t value;
        };
    }    // namespace

    region_data start_region(char const* new_region,
        std::size_t const thread_num, std::size_t const phase) noexcept
    {
        // clang-format off
#if defined(HPX_HAVE_STACKTRACES)
        TracyCZoneCS(ctx, static_cast<std::uint32_t>(thread_num),
            HPX_HAVE_THREAD_BACKTRACE_DEPTH, 1)
#else
        TracyCZoneC(ctx, static_cast<std::uint32_t>(thread_num), 1)
#endif
        TracyCZoneName(ctx, new_region, std::strlen(new_region))
        TracyCZoneValue(ctx, static_cast<std::uint32_t>(phase))

        tracy_context data;
        data.context = ctx;

        region_data& region = current_region();
        region_data const prev_region = region;

        region.name = new_region;
        region.data = data.value;
        region.color = static_cast<std::uint32_t>(thread_num);
        region.phase = static_cast<std::uint32_t>(phase);

        return prev_region;
        // clang-format on
    }

    region_data stop_region(region_data const& prev_region) noexcept
    {
        region_data const curr_region = current_region();

        current_region() = prev_region;
        if (curr_region.name != nullptr)
        {
            tracy_context data;
            data.value = curr_region.data;
            TracyCZoneEnd(data.context)
        }

        return curr_region;
    }

    char const* rename_region(char const* new_region) noexcept
    {
        if (auto& [name, value, _1_, _2_] = current_region(); name != nullptr)
        {
            // clang-format off
            char const* previous_name = name;
            name = new_region;

            tracy_context data;
            data.value = value;
            TracyCZoneName(data.context, new_region, std::strlen(new_region))
            return previous_name;
            // clang-format on
        }
        return nullptr;
    }
}    // namespace hpx::tracy

#endif
