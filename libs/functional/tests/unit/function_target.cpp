//  Copyright (c) 2016 Agustin Berge
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/functional/function.hpp>
#include <hpx/modules/testing.hpp>

#include <typeinfo>

struct foo
{
    int operator()()
    {
        return 0;
    }
    void operator()() const {}
};

int main()
{
    {
        hpx::util::function_nonser<int()> fun = foo();

        HPX_TEST(fun.target<foo>() != nullptr);
    }

    {
        hpx::util::function_nonser<int()> fun = foo();

        HPX_TEST(fun.target<foo const>() != nullptr);
    }

    return hpx::util::report_errors();
}
