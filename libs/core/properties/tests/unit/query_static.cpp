// Copyright (c) 2003-2020 Christopher M. Kohlhoff (chris at kohlhoff dot com)
// Copyright (c) 2021 Hartmut Kaiser
//
// SPDX-License-Identifier: BSL-1.0
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/modules/properties.hpp>
#include <hpx/modules/testing.hpp>

struct prop
{
    template <typename>
    static constexpr bool is_applicable_property_v = true;

    template <typename>
    static constexpr int static_query_v = 123;
};

struct object
{
};

int main()
{
    object o1 = {};
    int result1 = hpx::experimental::query(o1, prop());
    HPX_TEST(result1 == 123);

    object const o2 = {};
    int result2 = hpx::experimental::query(o2, prop());
    HPX_TEST(result2 == 123);

    constexpr object o3 = {};
    constexpr int result3 = hpx::experimental::query(o3, prop());
    HPX_TEST(result3 == 123);

    return hpx::util::report_errors();
}
