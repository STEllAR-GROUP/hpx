//  Copyright (c) 2015 Anton Bikineev
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <vector>

#include <hpx/runtime/serialization/serialize.hpp>

struct A
{
};

int main()
{
    std::vector<char> vector;
    {
        hpx::serialization::output_archive oar(vector);
        A a;
        oar << a;
    }
    {
        hpx::serialization::input_archive iar(vector);
        A a;
        iar >> a;
    }

    return 0;
}
