// Copyright (C) 2016 John Biddiscombe
//
// Distributed under the Boost Software License, Version 1.0. (See
// accompanying file LICENSE_1_0.txt or copy at

#ifndef HPT_PARCELSET_POLICIES_VERBS_PERFORMANCE_COUNTER_HPP
#define HPT_PARCELSET_POLICIES_VERBS_PERFORMANCE_COUNTER_HPP

#include <hpx/config/parcelport_verbs_defines.hpp>
#include <plugins/parcelport/verbs/rdma/rdma_logging.hpp>
//
#include <atomic>
#include <type_traits>
#include <iostream>

#ifdef HPX_PARCELPORT_VERBS_HAVE_PERFORMANCE_COUNTERS
#  define PERFORMANCE_COUNTER_ENABLED true
#else
#  define PERFORMANCE_COUNTER_ENABLED false
#endif

//
// This class is intended to provide a simple atomic counter that can be used as a
// performance counter, but that can be disabled at compile time so that it
// has no performance cost when not used. It is only to avoid a lot of #ifdef
// statements in user code that we collect everything in here and then provide
// the performance counter that will simply do nothing when disabled - but
// still allow code that uses the counters in arithmetic to compile.
//
namespace hpx {
namespace parcelset {
namespace policies {
namespace verbs
{
    template <bool B, typename T = void>
    using enable_if_t = typename std::enable_if<B, T>::type;

    template <typename T,
        bool enabled=PERFORMANCE_COUNTER_ENABLED,
        typename Enable = enable_if_t<std::is_integral<T>::value>
    >
    struct performance_counter {};

    // --------------------------------------------------------------------
    // specialization for performance counters Enabled
    // we provide an atomic<T> that can be incremented or added/subtracted to
    template <typename T>
    struct performance_counter<T, true>
    {
        performance_counter() : value_{T()} {}

        performance_counter(const T& init) : value_{init} {}

        inline operator T() { return value_; }

        inline T operator=(const T& x) { return value_ = x; }

        inline T operator++() { return ++value_; }

        inline T operator++(int x) { return (value_ += x); }

        inline T operator+=(const T& rhs) { return (value_ += rhs); }

        inline T operator--() { return --value_; }

        inline T operator--(int x) { return (value_ -= x); }

        inline T operator-=(const T& rhs) { return (value_ -= rhs); }

        friend std::ostream& operator<<(std::ostream& os,
            const performance_counter<T, true>& x)
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
        performance_counter() {}

        performance_counter(const T&) {}

        inline operator T() { return 0; }

        inline T operator=(const T&) { return 0; }

        inline T operator++() { return 0; }

        inline T operator++(int) { return 0; }

        inline T operator+=(const T&) { return 0; }

        inline T operator--() { return 0; }

        inline T operator--(int) { return 0; }

        inline T operator-=(const T&) { return 0; }

        friend std::ostream& operator<<(std::ostream& os,
            const performance_counter<T, false>& x)
        {
            os << "undefined";
            return os;
        }
    };
}}}}

#endif

