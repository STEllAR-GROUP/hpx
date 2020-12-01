//  Copyright (c) 2015 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/datastructures/tagged_pair.hpp>
#include <hpx/datastructures/tagged_tuple.hpp>
#include <hpx/hpx_init.hpp>
#include <hpx/modules/testing.hpp>

#include <type_traits>
#include <utility>

struct A
{
    A(int val = 0)
      : value_(val)
    {
    }
    int value_;
};

struct B
{
    B(int val = 0)
      : value_(val)
    {
    }
    int value_;
};

struct C
{
    C(int val = 0)
      : value_(val)
    {
    }
    int value_;
};

HPX_DEFINE_TAG_SPECIFIER(tag1)    // defines tag::tag1
HPX_DEFINE_TAG_SPECIFIER(tag2)    // defines tag::tag2
HPX_DEFINE_TAG_SPECIFIER(tag3)    // defines tag::tag3

void tagged_pair_test()
{
    typedef hpx::util::tagged_pair<tag::tag1(A), tag::tag2(B)> pair;

    static_assert(std::is_same<typename pair::first_type, A>::value, "");
    static_assert(std::is_same<typename pair::second_type, B>::value, "");

    {
        pair p;

        static_assert(std::is_same<decltype(p.first), A>::value, "");
        static_assert(std::is_same<decltype(p.second), B>::value, "");

        static_assert(
            std::is_same<typename std::decay<decltype(p.tag1())>::type,
                A>::value,
            "");
        static_assert(
            std::is_same<typename std::decay<decltype(p.tag2())>::type,
                B>::value,
            "");
    }

    {
        pair p(42, 43);

        HPX_TEST_EQ(p.tag1().value_, 42);
        HPX_TEST_EQ(p.tag2().value_, 43);
    }

    {
        pair p(42, 43);

        HPX_TEST_EQ(hpx::get<0>(p).value_, 42);
        HPX_TEST_EQ(hpx::get<1>(p).value_, 43);
    }

    {
        pair p = hpx::util::make_tagged_pair<tag::tag1, tag::tag2>(42, 43);

        HPX_TEST_EQ(p.tag1().value_, 42);
        HPX_TEST_EQ(p.tag2().value_, 43);
    }

    {
        pair p = hpx::util::make_tagged_pair<tag::tag1, tag::tag2>(
            std::make_pair(42, 43));

        HPX_TEST_EQ(p.tag1().value_, 42);
        HPX_TEST_EQ(p.tag2().value_, 43);
    }
}

void tagged_tuple_test()
{
    typedef hpx::util::tagged_tuple<tag::tag1(A), tag::tag2(B), tag::tag3(C)>
        tuple;

    {
        tuple t;

        static_assert(
            std::is_same<typename hpx::tuple_element<0, tuple>::type, A>::value,
            "");
        static_assert(
            std::is_same<typename hpx::tuple_element<1, tuple>::type, B>::value,
            "");
        static_assert(
            std::is_same<typename hpx::tuple_element<2, tuple>::type, C>::value,
            "");

        static_assert(
            std::is_same<typename std::decay<decltype(t.tag1())>::type,
                A>::value,
            "");
        static_assert(
            std::is_same<typename std::decay<decltype(t.tag2())>::type,
                B>::value,
            "");
        static_assert(
            std::is_same<typename std::decay<decltype(t.tag3())>::type,
                C>::value,
            "");
    }

    {
        tuple t(42, 43, 44);

        HPX_TEST_EQ(t.tag1().value_, 42);
        HPX_TEST_EQ(t.tag2().value_, 43);
        HPX_TEST_EQ(t.tag3().value_, 44);
    }

    {
        tuple t(42, 43, 44);

        HPX_TEST_EQ(hpx::get<0>(t).value_, 42);
        HPX_TEST_EQ(hpx::get<1>(t).value_, 43);
        HPX_TEST_EQ(hpx::get<2>(t).value_, 44);
    }

    {
        using hpx::util::make_tagged_tuple;
        tuple t =
            make_tagged_tuple<tag::tag1, tag::tag2, tag::tag3>(42, 43, 44);

        HPX_TEST_EQ(t.tag1().value_, 42);
        HPX_TEST_EQ(t.tag2().value_, 43);
        HPX_TEST_EQ(t.tag3().value_, 44);
    }

    {
        using hpx::util::make_tagged_tuple;
        tuple t = make_tagged_tuple<tag::tag1, tag::tag2, tag::tag3>(
            hpx::make_tuple(42, 43, 44));

        HPX_TEST_EQ(t.tag1().value_, 42);
        HPX_TEST_EQ(t.tag2().value_, 43);
        HPX_TEST_EQ(t.tag3().value_, 44);
    }
}

int main()
{
    tagged_pair_test();
    tagged_tuple_test();

    return hpx::util::report_errors();
}
