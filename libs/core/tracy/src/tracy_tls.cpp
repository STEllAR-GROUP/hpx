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
#include <mutex>
#include <string>
#include <unordered_set>

#include <tracy/TracyC.h>

///////////////////////////////////////////////////////////////////////////////
namespace hpx::tracy {

    namespace {

        // Global string interning for zone labels.
        // TracyCZoneName consumes raw pointers and can be called long after
        // the original source string's lifetime ended. Interning ensures
        // label pointers remain valid for the process lifetime.
        char const* intern_zone_label(char const* label, char const* fallback)
        {
            if (label == nullptr || label[0] == '\0')
            {
                label = fallback;
            }

            // A simple static lock and set for in-memory string interning.
            static std::unordered_set<std::string> interned_strings;
            static std::mutex interning_mutex;

            std::lock_guard<std::mutex> lock(interning_mutex);
            auto it = interned_strings.find(label);
            if (it != interned_strings.end())
            {
                return it->c_str();
            }

            constexpr std::size_t MAX_TRACY_LABELS = 4096;
            if (interned_strings.size() > MAX_TRACY_LABELS)
            {
                return fallback;
            }

            auto result = interned_strings.insert(label);
            return result.first->c_str();
        }

        // Store currently active Tracy zone
        region_data& current_region() noexcept
        {
            thread_local region_data region;
            return region;
        }

        // True while this OS thread is executing inside a Tracy fiber context.
        // rename_region (called by mark_event) must be a no-op in that case:
        // the zone ctx in TLS belongs to the OS-thread zone opened before
        // FiberEnter, but Tracy's internal zone stack has switched to the
        // fiber's stack — calling TracyCZoneName on the OS zone while inside
        // a fiber causes a "zone name destination doesn't match" crash.
        bool& in_fiber() noexcept
        {
            thread_local bool flag = false;
            return flag;
        }

        // This union is used to hide the Tracy context type from the user.
        union tracy_context
        {
            static_assert(sizeof(TracyCZoneCtx) == sizeof(std::uint64_t));

            TracyCZoneCtx context;
            std::uint64_t value;
        };

        // Store the zone ctx that was opened on the fiber's stack.
        // enter_fiber opens a zone (so the fiber track is visible in Tracy),
        // and leave_fiber closes it.
        // zone_name and color are cached so that suspend_fiber_zone can
        // reopen the running zone with the same label/color after the
        // task resumes from self_.yield().
        struct fiber_zone_data
        {
            std::uint64_t ctx_value = 0;
            char const* zone_name = "fiber";
            std::uint32_t color = 0;
            bool active = false;
        };

        fiber_zone_data& current_fiber_zone() noexcept
        {
            thread_local fiber_zone_data fz;
            return fz;
        }
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

    void set_in_fiber(bool value) noexcept
    {
        in_fiber() = value;
    }

    void start_fiber_zone(char const* zone_name, std::uint32_t color) noexcept
    {
        char const* safe_zone_name = "fiber";
        try
        {
            safe_zone_name = intern_zone_label(zone_name, "fiber");
        }
        catch (...)
        {
        }

        // clang-format off
        TracyCZoneC(ctx, color, 1)
        TracyCZoneName(ctx, safe_zone_name, std::strlen(safe_zone_name))
            // clang-format on

            tracy_context data;
        data.context = ctx;

        auto& fz = current_fiber_zone();
        fz.ctx_value = data.value;
        fz.zone_name = safe_zone_name;
        fz.color = color;
        fz.active = true;
    }

    void stop_fiber_zone() noexcept
    {
        auto& fz = current_fiber_zone();
        if (fz.active)
        {
            tracy_context data;
            data.value = fz.ctx_value;
            // clang-format off
            TracyCZoneEnd(data.context)
                // clang-format on
                fz.active = false;
            fz.ctx_value = 0;
        }
    }

    // Close the running fiber zone and open a "suspended" zone (grey) so
    // the fiber track shows a distinct bar during the suspension gap.
    // Called just before self_.yield() inside execution_agent::do_yield().
    // Only touches current_fiber_zone() — never calls stop_region() so the
    // OS-thread zone (current_region()) is completely untouched.
    void suspend_fiber_zone(char const* suspend_reason) noexcept
    {
        auto& fz = current_fiber_zone();
        if (!fz.active)
            return;

        // Close the running zone.
        {
            tracy_context data;
            data.value = fz.ctx_value;
            // clang-format off
            TracyCZoneEnd(data.context)
            // clang-format on
        }
        fz.active = false;

        // Open a grey "suspended" zone on the fiber stack.
        // 0xAAAAAA = medium grey, distinguishable from any worker-index color.
        constexpr std::uint32_t suspended_color = 0xAAAAAA;
        char const* safe_reason = "suspend";
        try
        {
            safe_reason = intern_zone_label(suspend_reason, "suspend");
        }
        catch (...)
        {
        }

        // clang-format off
        TracyCZoneC(sctx, suspended_color, 1)
        TracyCZoneName(sctx, safe_reason, std::strlen(safe_reason))
            // clang-format on

            tracy_context sdata;
        sdata.context = sctx;
        fz.ctx_value = sdata.value;
        fz.active = true;
        // zone_name and color are preserved from start_fiber_zone so that
        // resume_fiber_zone can restore the original running zone.
    }

    // Close the "suspended" zone and reopen the original running zone.
    // Called immediately after self_.yield() returns inside do_yield(),
    // i.e. when the task has been rescheduled onto a worker thread.
    // zone_name / color default to nullptr/0 which means "use the cached
    // values from start_fiber_zone".
    void resume_fiber_zone(char const* zone_name, std::uint32_t color) noexcept
    {
        auto& fz = current_fiber_zone();
        if (!fz.active)
            return;

        // Close the suspended (grey) zone.
        {
            tracy_context data;
            data.value = fz.ctx_value;
            // clang-format off
            TracyCZoneEnd(data.context)
            // clang-format on
        }
        fz.active = false;

        // Reopen the running zone with cached or supplied name/color.
        char const* name = (zone_name != nullptr) ? zone_name : fz.zone_name;
        char const* safe_name = "fiber";
        try
        {
            safe_name = intern_zone_label(name, "fiber");
        }
        catch (...)
        {
        }
        std::uint32_t col = (color != 0) ? color : fz.color;

        // clang-format off
        TracyCZoneC(rctx, col, 1)
        TracyCZoneName(rctx, safe_name, std::strlen(safe_name))
            // clang-format on

            tracy_context rdata;
        rdata.context = rctx;
        fz.ctx_value = rdata.value;
        fz.active = true;
    }

    char const* rename_region(char const* new_region) noexcept
    {
        // No-op inside a fiber context: the TLS zone ctx belongs to the
        // OS-thread zone, not the fiber's zone stack. Calling TracyCZoneName
        // here would corrupt Tracy's zone stack and cause an abort.
        if (in_fiber())
            return new_region;

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
