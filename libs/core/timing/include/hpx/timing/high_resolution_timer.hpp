//  Copyright (c) 2005-2023 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file high_resolution_timer.hpp

#pragma once

#include <hpx/config.hpp>
#include <hpx/timing/high_resolution_clock.hpp>

#include <cstdint>

namespace hpx::chrono {

    /// \brief high_resolution_timer is a timer object which measures
    ///        the elapsed time
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

        /// \brief returns the current time
        [[nodiscard]] static double now() noexcept
        {
            return static_cast<double>(take_time_stamp()) * 1e-9;
        }

        /// \brief restarts the timer
        void restart() noexcept
        {
            start_time_ = take_time_stamp();
        }

        /// \brief returns the elapsed time in seconds
        [[nodiscard]] double elapsed() const noexcept
        {
            return static_cast<double>(take_time_stamp() - start_time_) * 1e-9;
        }

        /// \brief returns the elapsed time in microseconds
        [[nodiscard]] std::int64_t elapsed_microseconds() const noexcept
        {
            return static_cast<std::int64_t>(
                static_cast<double>(take_time_stamp() - start_time_) * 1e-3);
        }

        /// \brief returns the elapsed time in nanoseconds
        [[nodiscard]] std::int64_t elapsed_nanoseconds() const noexcept
        {
            return static_cast<std::int64_t>(take_time_stamp() - start_time_);
        }

        /// \brief returns the estimated maximum value for \c elapsed()
        [[nodiscard]] static constexpr double elapsed_max() noexcept
        {
            return (hpx::chrono::high_resolution_clock::max)() * 1e-9;
        }

        /// \brief returns the estimated minimum value for \c elapsed()
        [[nodiscard]] static constexpr double elapsed_min() noexcept
        {
            return (hpx::chrono::high_resolution_clock::min)() * 1e-9;
        }

    protected:
        [[nodiscard]] static std::uint64_t take_time_stamp() noexcept
        {
            return hpx::chrono::high_resolution_clock::now();
        }

    private:
        std::uint64_t start_time_;
    };
}    // namespace hpx::chrono
