//  Taken from the Boost.Function library

//  Copyright Douglas Gregor 2001-2003.
//  Copyright 2013-2022 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Use, modification and
//  distribution is subject to the Boost Software License, Version
//  1.0. (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/functional/function.hpp>
#include <hpx/modules/testing.hpp>

#include <functional>

static int forty_two()
{
    return 42;
}

struct Seventeen
{
    int operator()() const
    {
        return 17;
    }
};

static void target_test()
{
    hpx::function<int()> f;

    f = &forty_two;
    HPX_TEST_EQ(*f.target<int (*)()>(), &forty_two);
    HPX_TEST(!f.target<Seventeen>());

    f = Seventeen();
    HPX_TEST(!f.target<int (*)()>());
    HPX_TEST(f.target<Seventeen>());

    Seventeen this_seventeen;
    f = this_seventeen;
    HPX_TEST(!f.target<int (*)()>());
    HPX_TEST(f.target<Seventeen>());
}

int main(int, char*[])
{
    target_test();

    return hpx::util::report_errors();
}
