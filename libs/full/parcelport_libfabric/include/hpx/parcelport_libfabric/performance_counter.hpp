// Copyright (C) 2016 John Biddiscombe
//
//  SPDX-License-Identifier: BSL-1.0
// Distributed under the Boost Software License, Version 1.0. (See
// accompanying file LICENSE_1_0.txt or copy at

#pragma once

#include <hpx/parcelport_libfabric/config/defines.hpp>
#include <hpx/parcelport_libfabric/parcelport_logging.hpp>
//
#include <atomic>
#include <iostream>
#include <type_traits>

#ifdef HPX_PARCELPORT_LIBFABRIC_HAVE_PERFORMANCE_COUNTERS
#define PERFORMANCE_COUNTER_ENABLED true
#else
#define PERFORMANCE_COUNTER_ENABLED false
#endif

//
// This class is intended to provide a simple atomic counter that can be used as a
// performance counter, but that can be disabled at compile time so that it
// has no performance cost when not used. It is only to avoid a lot of #ifdef
// statements in user code that we collect everything in here and then provide
// the performance counter that will simply do nothing when disabled - but
// still allow code that uses the counters in arithmetic to compile.
//
namespace hpx { namespace parcelset {

    template <typename T, bool enabled = PERFORMANCE_COUNTER_ENABLED,
        typename Enable = std::enable_if_t<std::is_integral<T>::value>>
    struct performance_counter
    {
    };

    // --------------------------------------------------------------------
    // specialization for performance counters Enabled
    // we provide an atomic<T> that can be incremented or added/subtracted to
    template <typename T>
    struct performance_counter<T, true>
    {
        performance_counter()
          : value_{T()}
        {
        }

        explicit performance_counter(const T& init)
          : value_{init}
        {
        }

        inline operator T() const
        {
            return value_;
        }

        inline T operator=(const T& x)
        {
            return value_ = x;
        }

        inline T operator++()
        {
            return ++value_;
        }

        inline T operator++(int x)
        {
            return (value_ += x);
        }

        inline T operator+=(const T& rhs)
        {
            return (value_ += rhs);
        }

        inline T operator--()
        {
            return --value_;
        }

        inline T operator--(int x)
        {
            return (value_ -= x);
        }

        inline T operator-=(const T& rhs)
        {
            return (value_ -= rhs);
        }

        friend std::ostream& operator<<(
            std::ostream& os, const performance_counter<T, true>& x)
        {
            os << x.value_;
            return os;
        }

        std::atomic<T> value_;
    };

    // --------------------------------------------------------------------
    // specialization for performance counters Disabled
    // just return dummy values so that arithmetic operations compile ok
    template <typename T>
    struct performance_counter<T, false>
    {
        constexpr performance_counter() = default;

        explicit constexpr performance_counter(const T&) {}

        inline constexpr operator T() const
        {
            return 0;
        }

        inline constexpr T operator=(const T&)
        {
            return 0;
        }

        inline constexpr T operator++()
        {
            return 0;
        }

        inline constexpr T operator++(int)
        {
            return 0;
        }

        inline constexpr T operator+=(const T&)
        {
            return 0;
        }

        inline constexpr T operator--()
        {
            return 0;
        }

        inline constexpr T operator--(int)
        {
            return 0;
        }

        inline constexpr T operator-=(const T&)
        {
            return 0;
        }

        friend std::ostream& operator<<(
            std::ostream& os, const performance_counter<T, false>&)
        {
            os << "undefined";
            return os;
        }
    };
}}    // namespace hpx::parcelset
