//  Copyright (c) 2005-2022 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/timing/high_resolution_clock.hpp>

#include <cstdint>

namespace hpx { namespace chrono {
    ///////////////////////////////////////////////////////////////////////////
    //
    //  high_resolution_timer
    //      A timer object measures elapsed time.
    //
    ///////////////////////////////////////////////////////////////////////////
    class high_resolution_timer
    {
    public:
        high_resolution_timer() noexcept
          : start_time_(take_time_stamp())
        {
        }

        enum class init
        {
            no_init
        };

        explicit constexpr high_resolution_timer(init) noexcept
          : start_time_(0)
        {
        }

        explicit constexpr high_resolution_timer(double t) noexcept
          : start_time_(static_cast<std::uint64_t>(t * 1e9))
        {
        }

        static double now() noexcept
        {
            return take_time_stamp() * 1e-9;
        }

        void restart() noexcept
        {
            start_time_ = take_time_stamp();
        }

        // return elapsed time in seconds
        double elapsed() const noexcept
        {
            return double(take_time_stamp() - start_time_) * 1e-9;
        }

        std::int64_t elapsed_microseconds() const noexcept
        {
            return std::int64_t(double(take_time_stamp() - start_time_) * 1e-3);
        }

        std::int64_t elapsed_nanoseconds() const noexcept
        {
            return std::int64_t(take_time_stamp() - start_time_);
        }

        // return estimated maximum value for elapsed()
        static constexpr double elapsed_max() noexcept
        {
            return (hpx::chrono::high_resolution_clock::max)() * 1e-9;
        }

        // return minimum value for elapsed()
        static constexpr double elapsed_min() noexcept
        {
            return (hpx::chrono::high_resolution_clock::min)() * 1e-9;
        }

    protected:
        static std::uint64_t take_time_stamp() noexcept
        {
            return hpx::chrono::high_resolution_clock::now();
        }

    private:
        std::uint64_t start_time_;
    };
}}    // namespace hpx::chrono
