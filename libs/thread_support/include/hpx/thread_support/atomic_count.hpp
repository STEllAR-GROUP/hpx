//  Copyright 2013 Peter Dimov
//  Copyright (c) 2016 Agustin Berge
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>

#include <atomic>

namespace hpx { namespace util {
    class atomic_count
    {
    public:
        HPX_NON_COPYABLE(atomic_count);

    public:
        explicit atomic_count(long value)
          : value_(value)
        {
        }

        atomic_count& operator=(long value)
        {
            value_.store(value, std::memory_order_relaxed);
            return *this;
        }

        long operator++()
        {
            return value_.fetch_add(1, std::memory_order_acq_rel) + 1;
        }

        long operator--()
        {
            return value_.fetch_sub(1, std::memory_order_acq_rel) - 1;
        }

        atomic_count& operator+=(long n)
        {
            value_.fetch_add(n, std::memory_order_acq_rel);
            return *this;
        }

        atomic_count& operator-=(long n)
        {
            value_.fetch_sub(n, std::memory_order_acq_rel);
            return *this;
        }

        operator long() const
        {
            return value_.load(std::memory_order_acquire);
        }

    private:
        std::atomic<long> value_;
    };
}}    // namespace hpx::util
