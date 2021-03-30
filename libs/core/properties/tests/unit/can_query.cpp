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
};

struct prop_not_applicable
{
};

struct prop_static
{
    template <typename>
    static constexpr int static_query_v = 123;
};

struct object_free
{
    friend constexpr int tag_invoke(
        hpx::experimental::query_t, object_free const&, prop)
    {
        return 123;
    }
};

struct object_member
{
    constexpr int query(prop) const
    {
        return 123;
    }
};

struct object_static
{
};

int main()
{
    using namespace hpx::experimental;

    static_assert(can_query_v<object_free, prop>, "");
    static_assert(can_query_v<object_free const, prop>, "");

    static_assert(can_query_v<object_member, prop>, "");
    static_assert(can_query_v<object_member const, prop>, "");

    static_assert(!can_query_v<object_free, prop_not_applicable>, "");
    static_assert(!can_query_v<object_free const, prop_not_applicable>, "");

    static_assert(!can_query_v<object_member, prop_not_applicable>, "");
    static_assert(!can_query_v<object_member const, prop_not_applicable>, "");

    static_assert(!can_query_v<object_static, prop_static>, "");
    static_assert(!can_query_v<object_static const, prop_static>, "");

    static_assert(!can_query_v<object_static, prop>, "");
    static_assert(!can_query_v<object_static const, prop>, "");

    static_assert(!can_query_v<object_static, prop_not_applicable>, "");
    static_assert(!can_query_v<object_static const, prop_not_applicable>, "");

    return hpx::util::report_errors();
}
