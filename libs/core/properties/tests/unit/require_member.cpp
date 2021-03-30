// Copyright (c) 2003-2020 Christopher M. Kohlhoff (chris at kohlhoff dot com)
// Copyright (c) 2021 Hartmut Kaiser
//
// SPDX-License-Identifier: BSL-1.0
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// 'require' is a dangerous Apple macro, silence inspect about this
// hpxinspect:noapple_macros:require

#include <hpx/modules/properties.hpp>
#include <hpx/modules/testing.hpp>

template <int>
struct prop
{
    template <typename>
    static constexpr bool is_applicable_property_v = true;

    static constexpr bool is_requirable = true;
};

template <int>
struct object
{
    template <int N>
    constexpr object<N> require(prop<N>) const
    {
        return object<N>();
    }
};

int main()
{
    object<1> o1 = {};
    object<2> o2 = hpx::experimental::require(o1, prop<2>());
    object<3> o3 = hpx::experimental::require(o1, prop<2>(), prop<3>());
    object<4> o4 =
        hpx::experimental::require(o1, prop<2>(), prop<3>(), prop<4>());
    (void) o2;
    (void) o3;
    (void) o4;

    object<1> const o5 = {};
    object<2> o6 = hpx::experimental::require(o5, prop<2>());
    object<3> o7 = hpx::experimental::require(o5, prop<2>(), prop<3>());
    object<4> o8 =
        hpx::experimental::require(o5, prop<2>(), prop<3>(), prop<4>());
    (void) o6;
    (void) o7;
    (void) o8;

    constexpr object<2> o9 = hpx::experimental::require(object<1>(), prop<2>());
    constexpr object<3> o10 =
        hpx::experimental::require(object<1>(), prop<2>(), prop<3>());
    constexpr object<4> o11 = hpx::experimental::require(
        object<1>(), prop<2>(), prop<3>(), prop<4>());
    (void) o9;
    (void) o10;
    (void) o11;

    return hpx::util::report_errors();
}
