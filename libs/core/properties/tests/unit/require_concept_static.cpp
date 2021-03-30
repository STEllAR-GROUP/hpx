// Copyright (c) 2003-2020 Christopher M. Kohlhoff (chris at kohlhoff dot com)
// Copyright (c) 2021 Hartmut Kaiser
//
// SPDX-License-Identifier: BSL-1.0
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/modules/properties.hpp>
#include <hpx/modules/testing.hpp>

template <int>
struct prop
{
    template <typename>
    static constexpr bool is_applicable_property_v = true;

    static constexpr bool is_requirable_concept = true;

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
    object<1> const& o2 = hpx::experimental::require_concept(o1, prop<1>());
    HPX_TEST(&o1 == &o2);
    (void) o2;

    object<1> const o3 = {};
    object<1> const& o4 = hpx::experimental::require_concept(o3, prop<1>());
    HPX_TEST(&o3 == &o4);
    (void) o4;

    constexpr object<1> o5 =
        hpx::experimental::require_concept(object<1>(), prop<1>());
    (void) o5;

    return hpx::util::report_errors();
}
