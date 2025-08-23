//  Copyright (c) 2018 Anton Bikineev
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/modules/serialization.hpp>
#include <hpx/modules/testing.hpp>

#include <vector>

struct A
{
    double a;
    int p;

    void serialize(int, unsigned)
    {
        // 3rd-party logic...
    }
};

template <class Ar>
void serialize(Ar& ar, A& a, unsigned)
{
    ar & a.a;
    ar & a.p;
}

int main()
{
    std::vector<char> vector;
    {
        hpx::serialization::output_archive oar(vector);
        A a{2., 4};
        oar << a;
    }

    {
        A a;
        hpx::serialization::input_archive iar(vector);
        iar >> a;
        HPX_TEST_EQ(a.a, 2.);
        HPX_TEST_EQ(a.p, 4);
    }

    return 0;
}
