//  Copyright (c) 2007-2023 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file high_resolution_clock.hpp

#pragma once

#include <hpx/config.hpp>

#if defined(__bgq__)
#include <hwi/include/bqc/A2_inlines.h>
#endif

#include <chrono>
#include <cstdint>

namespace hpx::chrono {

    /// \brief Class \c hpx::chrono::high_resolution_clock represents the clock
    ///        with the smallest tick period provided by the implementation. It
    ///        may be an alias of \c std::chrono::system_clock or
    ///        \c std::chrono::steady_clock, or a third, independent clock.
    ///        \c hpx::chrono::high_resolution_clock meets the requirements of
    ///        \a TrivialClock.
    struct high_resolution_clock
    {
        // This function returns a tick count with a resolution (not
        // precision!) of 1 ns.
        /// returns a \c std::chrono::time_point representing the current value
        /// of the clock
        [[nodiscard]] static std::uint64_t now() noexcept
        {
#if defined(__bgq__)
            return GetTimeBase();
#else
            std::chrono::nanoseconds const ns =
                std::chrono::steady_clock::now().time_since_epoch();
            return static_cast<std::uint64_t>(ns.count());
#endif
        }

        // This function returns the smallest representable time unit as
        // returned by this clock.
        [[nodiscard]] static constexpr std::uint64_t(min)() noexcept
        {
            using duration_values =
                std::chrono::duration_values<std::chrono::nanoseconds>;
            return (duration_values::min)().count();
        }

        // This function returns the largest representable time unit as
        // returned by this clock.
        [[nodiscard]] static constexpr std::uint64_t(max)() noexcept
        {
            using duration_values =
                std::chrono::duration_values<std::chrono::nanoseconds>;
            return (duration_values::max)().count();
        }
    };
}    // namespace hpx::chrono
