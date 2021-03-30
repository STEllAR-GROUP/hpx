//  Copyright (c) 2020 ETH Zurich
//  Copyright (c) 2021 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/modules/functional.hpp>
#include <hpx/modules/properties.hpp>
#include <hpx/modules/testing.hpp>

#include <type_traits>

///////////////////////////////////////////////////////////////////////////////
struct property1
{
    static constexpr bool is_requirable_concept = true;

    template <typename T>
    static constexpr bool is_applicable_property_v = true;

    int v = 0;
};

struct type1
{
    property1 p1{};
};

type1 tag_invoke(
    hpx::experimental::require_concept_t, type1 const& t, property1 p)
{
    auto tt = t;
    tt.p1 = p;
    return tt;
}

///////////////////////////////////////////////////////////////////////////////
struct type2;

template <typename T>
struct is_type2 : std::is_same<type2, std::decay_t<T>>
{
};

struct type3;

template <typename T>
struct is_type3 : std::is_same<type3, std::decay_t<T>>
{
};

///////////////////////////////////////////////////////////////////////////////
struct property2
{
    static constexpr bool is_requirable_concept = true;

    template <typename T>
    static constexpr bool is_applicable_property_v =
        is_type2<T>::value || is_type3<T>::value;

    int v = 0;
};

struct type2
{
    property2 p2{};
};

type2 tag_invoke(
    hpx::experimental::require_concept_t, type2 const& t, property2 p)
{
    auto tt = t;
    tt.p2 = p;
    return tt;
}

///////////////////////////////////////////////////////////////////////////////
struct property3
{
    static constexpr bool is_requirable_concept = true;

    template <typename T>
    static constexpr bool is_applicable_property_v = is_type3<T>::value;

    int v = 0;
};

struct type3
{
    property3 p3{};

    type3 require_concept(property3 p) const
    {
        auto tt = *this;
        tt.p3 = p;
        return tt;
    }

    type3& operator=(property2 p2)
    {
        p3.v = p2.v;
        return *this;
    }
};

type3 tag_invoke(
    hpx::experimental::require_concept_t, type3 const& t, property2 p)
{
    auto tt = t;
    tt = p;
    return tt;
}

///////////////////////////////////////////////////////////////////////////////
struct type4;

template <typename T>
struct is_type4 : std::is_same<type4, std::decay_t<T>>
{
};

struct property4
{
    static constexpr bool is_requirable_concept = true;

    template <typename T>
    static constexpr bool is_applicable_property_v = is_type4<T>::value;

    template <typename T>
    static constexpr bool static_query_v = is_type4<T>::value;

    static constexpr bool value()
    {
        return true;
    }

    int v = 0;
};

struct type4
{
    property4 p4{};
};

type4 tag_invoke(
    hpx::experimental::require_concept_t, type4 const& t, property4 p)
{
    auto tt = t;
    tt.p4 = p;
    return tt;
}

///////////////////////////////////////////////////////////////////////////////
int main()
{
    // This property is accessed through a free function
    {
        static_assert(
            hpx::experimental::can_require_concept_v<type1, property1>,
            "Should be requirable");
        static_assert(hpx::is_invocable<hpx::experimental::require_concept_t,
                          type1, property1>::value,
            "Should be invocable");

        static_assert(
            !hpx::experimental::can_require_concept_v<type2, property1>,
            "Should not be requirable");
        static_assert(!hpx::is_invocable<hpx::experimental::require_concept_t,
                          type2, property1>::value,
            "Should not be invocable");

        static_assert(
            !hpx::experimental::can_require_concept_v<type3, property1>,
            "Should not be requirable");
        static_assert(!hpx::is_invocable<hpx::experimental::require_concept_t,
                          type3, property1>::value,
            "Should not be invocable");

        static_assert(
            !hpx::experimental::can_require_concept_v<type4, property1>,
            "Should not be requirable");
        static_assert(!hpx::is_invocable<hpx::experimental::require_concept_t,
                          type4, property1>::value,
            "Should not be invocable");

        type1 t1_1{};

        type1 t1_2 = hpx::experimental::require_concept(t1_1, property1{1});
        HPX_TEST_EQ(t1_1.p1.v, 0);
        HPX_TEST_EQ(t1_2.p1.v, 1);

        property1 p1_1{2};
        type1 t1_3 = hpx::experimental::require_concept(t1_2, p1_1);
        HPX_TEST_EQ(t1_2.p1.v, 1);
        HPX_TEST_EQ(t1_3.p1.v, 2);

        property1 const p1_2{3};
        type1 t1_4 = hpx::experimental::require_concept(t1_2, p1_2);
        HPX_TEST_EQ(t1_3.p1.v, 2);
        HPX_TEST_EQ(t1_4.p1.v, 3);
    }

    // This property is accessed through a member function
    {
        static_assert(
            !hpx::experimental::can_require_concept_v<type1, property2>,
            "Should not be requirable");
        static_assert(!hpx::is_invocable<hpx::experimental::require_concept_t,
                          type1, property2>::value,
            "Should not be invocable");

        static_assert(
            hpx::experimental::can_require_concept_v<type2, property2>,
            "Should be requirable");
        static_assert(hpx::is_invocable<hpx::experimental::require_concept_t,
                          type2, property2>::value,
            "Should be invocable");

        static_assert(
            hpx::experimental::can_require_concept_v<type3, property2>,
            "Should be requirable");
        static_assert(hpx::is_invocable<hpx::experimental::require_concept_t,
                          type3, property2>::value,
            "Should be invocable");

        static_assert(
            !hpx::experimental::can_require_concept_v<type4, property2>,
            "Should not be requirable");
        static_assert(!hpx::is_invocable<hpx::experimental::require_concept_t,
                          type4, property2>::value,
            "Should not be invocable");

        type2 t2_1{};

        type2 t2_2 = hpx::experimental::require_concept(t2_1, property2{1});
        HPX_TEST_EQ(t2_1.p2.v, 0);
        HPX_TEST_EQ(t2_2.p2.v, 1);

        property2 p2_1{2};
        type2 t2_3 = hpx::experimental::require_concept(t2_2, p2_1);
        HPX_TEST_EQ(t2_2.p2.v, 1);
        HPX_TEST_EQ(t2_3.p2.v, 2);

        property2 const p2_2{3};
        type2 t2_4 = hpx::experimental::require_concept(t2_2, p2_2);
        HPX_TEST_EQ(t2_3.p2.v, 2);
        HPX_TEST_EQ(t2_4.p2.v, 3);
    }

    {
        static_assert(
            !hpx::experimental::can_require_concept_v<type1, property3>,
            "Should not be requirable");
        static_assert(!hpx::is_invocable<hpx::experimental::require_concept_t,
                          type1, property3>::value,
            "Should not be invocable");

        static_assert(
            !hpx::experimental::can_require_concept_v<type2, property3>,
            "Should not be requirable");
        static_assert(!hpx::is_invocable<hpx::experimental::require_concept_t,
                          type2, property3>::value,
            "Should not be invocable");

        static_assert(
            hpx::experimental::can_require_concept_v<type3, property3>,
            "Should be requirable");
        static_assert(hpx::is_invocable<hpx::experimental::require_concept_t,
                          type3, property3>::value,
            "Should be invocable");

        static_assert(
            !hpx::experimental::can_require_concept_v<type4, property3>,
            "Should not be requirable");
        static_assert(!hpx::is_invocable<hpx::experimental::require_concept_t,
                          type4, property3>::value,
            "Should not be invocable");

        type3 t3_1{};

        type3 t3_2 = hpx::experimental::require_concept(t3_1, property3{1});
        HPX_TEST_EQ(t3_1.p3.v, 0);
        HPX_TEST_EQ(t3_2.p3.v, 1);

        property3 p3_1{2};
        type3 t3_3 = hpx::experimental::require_concept(t3_2, p3_1);
        HPX_TEST_EQ(t3_2.p3.v, 1);
        HPX_TEST_EQ(t3_3.p3.v, 2);

        property3 const p3_2{3};
        type3 t3_4 = hpx::experimental::require_concept(t3_3, p3_2);
        HPX_TEST_EQ(t3_3.p3.v, 2);
        HPX_TEST_EQ(t3_4.p3.v, 3);

        type3 t3_5 = hpx::experimental::require_concept(t3_4, property2{1});
        HPX_TEST_EQ(t3_4.p3.v, 3);
        HPX_TEST_EQ(t3_5.p3.v, 1);

        property2 p2_1{2};
        type3 t3_6 = hpx::experimental::require_concept(t3_5, p2_1);
        HPX_TEST_EQ(t3_5.p3.v, 1);
        HPX_TEST_EQ(t3_6.p3.v, 2);

        property2 const p2_2{3};
        type3 t3_7 = hpx::experimental::require_concept(t3_6, p2_2);
        HPX_TEST_EQ(t3_6.p3.v, 2);
        HPX_TEST_EQ(t3_7.p3.v, 3);
    }

    {
        static_assert(
            !hpx::experimental::can_require_concept_v<type1, property4>,
            "Should not be requirable");
        static_assert(!hpx::is_invocable<hpx::experimental::require_concept_t,
                          type1, property4>::value,
            "Should not be invocable");

        static_assert(
            !hpx::experimental::can_require_concept_v<type2, property4>,
            "Should not be requirable");
        static_assert(!hpx::is_invocable<hpx::experimental::require_concept_t,
                          type2, property4>::value,
            "Should not be invocable");

        static_assert(
            !hpx::experimental::can_require_concept_v<type3, property4>,
            "Should not be requirable");
        static_assert(!hpx::is_invocable<hpx::experimental::require_concept_t,
                          type3, property4>::value,
            "Should not be invocable");

        static_assert(
            hpx::experimental::can_require_concept_v<type4, property4>,
            "Should be requirable");
        static_assert(hpx::is_invocable<hpx::experimental::require_concept_t,
                          type4, property4>::value,
            "Should be invocable");

        type4 t4_1{};

        type4 t4_2 = hpx::experimental::require_concept(t4_1, property4{1});
        HPX_TEST_EQ(t4_1.p4.v, 0);
        HPX_TEST_EQ(t4_2.p4.v, 0);

        property4 p4_1{2};
        type4 t4_3 = hpx::experimental::require_concept(t4_2, p4_1);
        HPX_TEST_EQ(t4_2.p4.v, 0);
        HPX_TEST_EQ(t4_3.p4.v, 0);

        property4 const p4_2{4};
        type4 t4_4 = hpx::experimental::require_concept(t4_3, p4_2);
        HPX_TEST_EQ(t4_3.p4.v, 0);
        HPX_TEST_EQ(t4_4.p4.v, 0);
    }

    return hpx::util::report_errors();
}
