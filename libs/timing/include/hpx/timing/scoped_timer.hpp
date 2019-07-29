////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2016 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#ifndef HPX_UTIL_SCOPED_TIMER_HPP
#define HPX_UTIL_SCOPED_TIMER_HPP

#include <hpx/timing/high_resolution_clock.hpp>

#include <cstdint>

namespace hpx { namespace util {
    template <typename T>
    struct scoped_timer
    {
        scoped_timer(T& t, bool enabled = true)
          : started_at_(enabled ? hpx::util::high_resolution_clock::now() : 0)
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
                *t_ += (hpx::util::high_resolution_clock::now() - started_at_);
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

        bool enabled() const noexcept
        {
            return t_ != nullptr;
        }

    private:
        std::uint64_t started_at_;
        T* t_;
    };
}}

#endif
