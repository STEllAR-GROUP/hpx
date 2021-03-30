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

    template <typename>
    static constexpr bool static_query_v = true;

    static constexpr bool value()
    {
        return true;
    }
};

template <int>
struct object
{
};

int main()
{
    object<1> o1 = {};
    object<1> const& o2 = hpx::experimental::require(o1, prop<1>());
    HPX_TEST(&o1 == &o2);
    object<1> const& o3 = hpx::experimental::require(o1, prop<1>(), prop<1>());
    HPX_TEST(&o1 == &o3);
    object<1> const& o4 =
        hpx::experimental::require(o1, prop<1>(), prop<1>(), prop<1>());
    HPX_TEST(&o1 == &o4);

    object<1> const o5 = {};
    object<1> const& o6 = hpx::experimental::require(o5, prop<1>());
    HPX_TEST(&o5 == &o6);
    object<1> const& o7 = hpx::experimental::require(o5, prop<1>(), prop<1>());
    HPX_TEST(&o5 == &o7);
    object<1> const& o8 =
        hpx::experimental::require(o5, prop<1>(), prop<1>(), prop<1>());
    HPX_TEST(&o5 == &o8);

    constexpr object<1> o9 = hpx::experimental::require(object<1>(), prop<1>());
    constexpr object<1> o10 =
        hpx::experimental::require(object<1>(), prop<1>(), prop<1>());
    constexpr object<1> o11 = hpx::experimental::require(
        object<1>(), prop<1>(), prop<1>(), prop<1>());
    (void) o9;
    (void) o10;
    (void) o11;

    return hpx::util::report_errors();
}
