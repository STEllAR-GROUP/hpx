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
};

template <int>
struct prop_not_applicable
{
    static constexpr bool is_requirable_concept = true;
};

template <int>
struct prop_unsupported
{
};

template <int>
struct prop_static
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
struct object_free
{
    template <int N>
    friend constexpr object_free<N> tag_invoke(
        hpx::experimental::require_concept_t, object_free const&, prop<N>)
    {
        return object_free<N>();
    }

    template <int N>
    friend constexpr object_free<N> tag_invoke(
        hpx::experimental::require_concept_t, object_free const&,
        prop_not_applicable<N>)
    {
        return object_free<N>();
    }
};

template <int>
struct object_member
{
    template <int N>
    constexpr object_member<N> require_concept(prop<N>) const
    {
        return object_member<N>();
    }

    template <int N>
    constexpr object_member<N> require_concept(prop_not_applicable<N>) const
    {
        return object_member<N>();
    }
};

template <int>
struct object_static
{
};

int main()
{
    using namespace hpx::experimental;

    static_assert(can_require_concept_v<object_free<1>, prop<2>>, "");
    static_assert(can_require_concept_v<object_free<1> const, prop<2>>, "");

    static_assert(can_require_concept_v<object_member<1>, prop<2>>, "");
    static_assert(can_require_concept_v<object_member<1> const, prop<2>>, "");

    static_assert(can_require_concept_v<object_static<1>, prop_static<1>>, "");
    static_assert(
        can_require_concept_v<object_static<1> const, prop_static<1>>, "");

    static_assert(
        !can_require_concept_v<object_free<1>, prop_not_applicable<2>>, "");
    static_assert(
        !can_require_concept_v<object_free<1> const, prop_not_applicable<2>>,
        "");

    static_assert(
        !can_require_concept_v<object_member<1>, prop_not_applicable<2>>, "");
    static_assert(
        !can_require_concept_v<object_member<1> const, prop_not_applicable<2>>,
        "");

    static_assert(
        !can_require_concept_v<object_static<1>, prop_not_applicable<1>>, "");
    static_assert(
        !can_require_concept_v<object_static<1> const, prop_not_applicable<1>>,
        "");

    static_assert(
        !can_require_concept_v<object_static<1>, prop_unsupported<2>>, "");
    static_assert(
        !can_require_concept_v<object_static<1> const, prop_unsupported<2>>,
        "");

    return hpx::util::report_errors();
}
