//  Copyright (c) 2019 Jan Melech
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/runtime/serialization/serialize.hpp>
#include <hpx/runtime/serialization/optional.hpp>
#include <hpx/runtime/serialization/brace_initializable.hpp>

#include <hpx/runtime/serialization/input_archive.hpp>
#include <hpx/runtime/serialization/output_archive.hpp>

#include <hpx/util/lightweight_test.hpp>

#include <vector>
#include <tuple>
#include <string>

struct A
{
    std::string str;
    double floating_number;
    int int_number;
};

bool operator==(const A& a1, const A& a2)
{
    return std::tie(a1.str, a1.floating_number, a1.int_number)
        == std::tie(a2.str, a2.floating_number, a2.int_number);
}

struct B
{
    A a;
    char sign;
};

bool operator==(const B& b1, const B& b2)
{
    return std::tie(b1.a, b1.sign)
        == std::tie(b2.a, b2.sign);
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
