//  Copyright (c) 2019 Jan Melech
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>

#if defined(HPX_HAVE_CXX17_STRUCTURED_BINDINGS)
#include <hpx/modules/serialization.hpp>
#include <hpx/modules/testing.hpp>

#include <string>
#include <tuple>
#include <vector>

struct A
{
    std::string str;
    double floating_number;
    int int_number;
};

static_assert(hpx::traits::is_brace_constructible<A, 3>(),
    "hpx::traits::is_brace_constructible<A, 3>()");
static_assert(!hpx::traits::is_brace_constructible<A, 4>(),
    "!hpx::traits::is_brace_constructible<A, 4>()");
static_assert(!hpx::traits::is_paren_constructible<A, 3>(),
    "!hpx::traits::is_paren_constructible<A, 3>()");

static_assert(hpx::traits::detail::arity<A>().value == 3,
    "hpx::traits::detail::arity<A>() == size<3>{}");
static_assert(hpx::serialization::has_struct_serialization<A>::value,
    "has_struct_serialization<A>::value");
static_assert(!hpx::serialization::has_serialize_adl<A>::value,
    "!has_serialize_adl<A>::value");

bool operator==(const A& a1, const A& a2)
{
    return std::tie(a1.str, a1.floating_number, a1.int_number) ==
        std::tie(a2.str, a2.floating_number, a2.int_number);
}

struct B
{
    A a;
    char sign;
};

static_assert(hpx::traits::is_brace_constructible<B, 2>(),
    "hpx::traits::is_brace_constructible<B, 2>()");
static_assert(!hpx::traits::is_brace_constructible<B, 3>(),
    "!hpx::traits::is_brace_constructible<B, 3>()");
static_assert(!hpx::traits::is_paren_constructible<B, 2>(),
    "!hpx::traits::is_paren_constructible<B, 2>()");

static_assert(hpx::traits::detail::arity<B>().value == 2,
    "hpx::traits::detail::arity<B>() == size<2>{}");
static_assert(hpx::serialization::has_struct_serialization<B>::value,
    "has_struct_serialization<B>::value");
static_assert(!hpx::serialization::has_serialize_adl<B>::value,
    "!has_serialize_adl<B>::value");

bool operator==(const B& b1, const B& b2)
{
    return std::tie(b1.a, b1.sign) == std::tie(b2.a, b2.sign);
}

int main()
{
    std::vector<char> buf;
    hpx::serialization::output_archive oar(buf);
    hpx::serialization::input_archive iar(buf);

    {
        A a{"test_string", 1234.8281, -1919};
        oar << a;
        A deserialized_a;
        iar >> deserialized_a;

        HPX_TEST(a == deserialized_a);
    }

    {
        A a{"test_string", 1234.8281, -1919};
        B b{a, 'u'};
        oar << b;
        B deserialized_b;
        iar >> deserialized_b;

        HPX_TEST(b == deserialized_b);
    }

    return hpx::util::report_errors();
}
#else
int main()
{
    return 0;
}
#endif
