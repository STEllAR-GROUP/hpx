////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2016 Thomas Heller
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#pragma once

#include <hpx/timing/high_resolution_clock.hpp>

#include <cstdint>

namespace hpx::util {

    template <typename T>
    struct scoped_timer
    {
        explicit scoped_timer(T& t, bool enabled = true) noexcept
          : started_at_(enabled ? hpx::chrono::high_resolution_clock::now() : 0)
          , t_(enabled ? &t : nullptr)
        {
        }

        scoped_timer(scoped_timer const&) = delete;
        scoped_timer(scoped_timer&& rhs) noexcept
          : started_at_(rhs.started_at_)
          , t_(rhs.t_)
        {
            rhs.t_ = nullptr;
        }

        ~scoped_timer()
        {
            if (enabled())
            {
                *t_ +=
                    (hpx::chrono::high_resolution_clock::now() - started_at_);
            }
        }

        scoped_timer& operator=(scoped_timer const& rhs) = delete;
        scoped_timer& operator=(scoped_timer&& rhs) noexcept
        {
            started_at_ = rhs.started_at_;
            t_ = rhs.t_;
            rhs.t_ = nullptr;
            return *this;
        }

        [[nodiscard]] constexpr bool enabled() const noexcept
        {
            return t_ != nullptr;
        }

    private:
        std::uint64_t started_at_;
        T* t_;
    };
}    // namespace hpx::util
